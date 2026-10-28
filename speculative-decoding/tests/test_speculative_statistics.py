"""Seeded distribution tests for speculative decoding against exact toy oracles.

Run with: python -m pytest -q -s speculative-decoding/tests/test_speculative_statistics.py
No model downloads or external statistical packages are required.
"""

from __future__ import annotations

import math

import pytest
import torch

from .decoding_fixtures import ToyLanguageModel
from src.decoding.speculative import SpeculativeDecoder


def _temperature_probs(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Exact softmax(log(p)/temperature), including zero-probability entries."""
    logits = torch.where(probs > 0, probs.log(), torch.full_like(probs, -torch.inf))
    return torch.softmax(logits / temperature, dim=-1)


def _top_k_probs(probs: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    logits = torch.where(probs > 0, probs.log(), torch.full_like(probs, -torch.inf))
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[torch.topk(logits, k).indices] = True
    return torch.softmax(torch.where(keep, logits, -torch.inf) / temperature, dim=-1)


def _assert_multinomial_close(counts: torch.Tensor, expected: torch.Tensor, *, label: str):
    """Simultaneous Hoeffding bound over all categories (family-wise alpha=1e-6)."""
    trials = int(counts.sum())
    expected = expected.detach().cpu().to(torch.float64)
    observed = counts.to(torch.float64) / trials
    epsilon = math.sqrt(math.log(2 * expected.numel() / 1e-6) / (2 * trials))
    max_error = (observed - expected).abs().max().item()
    print(
        f"{label}: n={trials}, counts={counts.tolist()}, "
        f"observed={observed.tolist()}, expected={expected.tolist()}, "
        f"max_abs_error={max_error:.5f}, hoeffding_epsilon={epsilon:.5f}"
    )
    assert max_error <= epsilon, (
        f"{label}: max absolute frequency error {max_error:.5f} exceeds "
        f"simultaneous Hoeffding bound {epsilon:.5f}; "
        f"counts={counts.tolist()}, observed={observed.tolist()}, "
        f"expected={expected.tolist()}"
    )


@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_acceptance_and_rejection_frequencies_match_min_p_over_q(top_k, temperature):
    """Acceptance is min(1,p/q) under both truncated and full draft support."""
    torch.manual_seed(713 + top_k * 10 + round(temperature * 10))
    n = 12_000
    draft_raw = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float64)
    target_raw = torch.tensor([0.20, 0.30, 0.30, 0.20], dtype=torch.float64)
    q = _top_k_probs(draft_raw, top_k, temperature)
    p = _temperature_probs(target_raw, temperature)
    target = ToyLanguageModel(target_raw)
    draft = ToyLanguageModel(draft_raw)
    decoder = SpeculativeDecoder(draft, target, k=1, top_k=top_k, temperature=temperature)

    proposed = torch.multinomial(q, n, replacement=True)
    accepted = torch.zeros(n, dtype=torch.bool)
    # Mark the decision branch directly: returned counters include the replacement token.
    resample_calls = []
    decoder._resample_rejected = lambda *args: (resample_calls.append(True) or torch.tensor([3]))
    for trial, token in enumerate(proposed.tolist()):
        resamples_before = len(resample_calls)
        decoder.acceptance_step(
            torch.tensor([[0]]),
            torch.tensor([[token]]),
            q[token : token + 1],
            p[token : token + 1],
            eos_token_id=None,
        )
        accepted[trial] = len(resample_calls) == resamples_before

    expected_acceptance = torch.minimum(torch.ones_like(q), p / q.clamp_min(1e-300))
    expected_acceptance = (q * expected_acceptance).sum()
    expected = torch.tensor([expected_acceptance, 1 - expected_acceptance])
    counts = torch.tensor([accepted.sum().item(), (~accepted).sum().item()])
    _assert_multinomial_close(
        counts, expected, label=f"acceptance/rejection top_k={top_k} temperature={temperature}"
    )
    # A correct aggregate rate could hide wrong decisions for individual tokens.
    for token in range(q.numel()):
        selected = proposed == token
        if not selected.any():
            continue
        token_accepted = accepted[selected]
        ratio = min(1.0, (p[token] / q[token]).item())
        _assert_multinomial_close(
            torch.tensor([token_accepted.sum(), (~token_accepted).sum()]),
            torch.tensor([ratio, 1 - ratio]),
            label=f"conditional acceptance token={token} top_k={top_k} temperature={temperature}",
        )


def test_conditional_rejection_samples_from_top_k_residual_distribution():
    """Conditional rejection must subtract the proposal's *truncated* q(x)."""
    torch.manual_seed(20261102)
    n = 15_000
    draft_raw = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float64)
    target_raw = torch.tensor([0.0, 0.0, 0.30, 0.70], dtype=torch.float64)
    q = _top_k_probs(draft_raw, k=2, temperature=1.0)
    p = target_raw
    residual = (p - q).clamp_min(0)
    residual /= residual.sum()
    draft = ToyLanguageModel(draft_raw)
    target = ToyLanguageModel(target_raw)
    decoder = SpeculativeDecoder(draft, target, k=1, top_k=2, temperature=1.0)

    proposed = torch.multinomial(q, n, replacement=True)
    rejected_resamples = []
    for token in proposed.tolist():
        new_ids, _, _, _ = decoder.acceptance_step(
            torch.tensor([[0]]),
            torch.tensor([[token]]),
            q[token : token + 1],
            p[token : token + 1],
            eos_token_id=None,
        )
        # p(x)=0 for every top-k draft candidate, so rejection is certain.
        # Read the replacement from the real acceptance_step output.
        assert new_ids[0, -1].item() in (2, 3)
        rejected_resamples.append(new_ids[0, -1].item())

    counts = torch.bincount(torch.tensor(rejected_resamples), minlength=4)
    _assert_multinomial_close(
        counts,
        residual,
        label=f"top_k residual conditioned on rejection (rejections={len(rejected_resamples)})",
    )


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_generation_matches_target_first_token_and_two_token_joint(k, top_k, temperature):
    """Distinct target/draft laws; check full real generation against target joint law."""
    torch.manual_seed(9127 + top_k * 10 + round(temperature * 10))
    n = 12_000
    draft_row0 = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float64)
    target_row0 = torch.tensor([0.20, 0.30, 0.30, 0.20], dtype=torch.float64)
    target_by_token = torch.stack(
        [
            target_row0,
            torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float64),
            torch.tensor([0.40, 0.30, 0.20, 0.10], dtype=torch.float64),
            torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64),
        ]
    )
    draft_by_token = torch.stack(
        [draft_row0,
         torch.tensor([0.45, 0.25, 0.20, 0.10], dtype=torch.float64),
         torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float64),
         torch.tensor([0.40, 0.30, 0.20, 0.10], dtype=torch.float64)]
    )
    draft = ToyLanguageModel(draft_by_token)
    target = ToyLanguageModel(target_by_token)
    decoder = SpeculativeDecoder(draft, target, k=k, top_k=top_k, temperature=temperature)
    first_counts = torch.zeros(4, dtype=torch.long)
    joint_counts = torch.zeros(16, dtype=torch.long)
    for _ in range(n):
        token_ids, _, _ = decoder.generate("toy", max_new_tokens=2)
        # Measure the first generated prefix, even when k>remaining_budget
        # exposes the independent hard-cap defect covered in the edge suite.
        first, second = token_ids[1:3].tolist()
        first_counts[first] += 1
        joint_counts[first * 4 + second] += 1

    target_first = _temperature_probs(target_row0, temperature)
    target_transitions = torch.stack(
        [_temperature_probs(row, temperature) for row in target_by_token]
    )
    target_joint = (target_first[:, None] * target_transitions).reshape(-1)
    # Report both independent oracles even if the first-token check detects a bias.
    failures = []
    for counts, expected, label in (
        (first_counts, target_first,
         f"first token k={k} top_k={top_k} temperature={temperature}"),
        (joint_counts, target_joint,
         f"two-token target joint k={k} top_k={top_k} temperature={temperature}"),
    ):
        try:
            _assert_multinomial_close(counts, expected, label=label)
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, "\\n".join(failures)


def test_disjoint_support_and_identical_target_draft_laws():
    """Exercise zero overlap and p=q edge cases in the actual generation loop."""
    torch.manual_seed(331)
    n = 8_000
    draft_raw = torch.tensor([0.65, 0.35, 0.0, 0.0], dtype=torch.float64)
    target_raw = torch.tensor([0.0, 0.0, 0.40, 0.60], dtype=torch.float64)
    decoder = SpeculativeDecoder(
        ToyLanguageModel(draft_raw), ToyLanguageModel(target_raw),
        k=1, top_k=4, temperature=1.0,
    )
    first_counts = torch.zeros(4, dtype=torch.long)
    for _ in range(n):
        token_ids, _, _ = decoder.generate("toy", max_new_tokens=1)
        first = token_ids[-1].item()
        first_counts[first] += 1
    _assert_multinomial_close(first_counts, target_raw, label="disjoint-support target output")

    same_raw = torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float64)
    same = ToyLanguageModel(same_raw)
    same_decoder = SpeculativeDecoder(same, same, k=1, top_k=4, temperature=1.0)
    same_counts = torch.zeros(4, dtype=torch.long)
    for _ in range(n):
        token_ids, _, stats = same_decoder.generate("toy", max_new_tokens=1)
        same_counts[token_ids[-1].item()] += 1
        assert stats.rejected_tokens == 0
    _assert_multinomial_close(same_counts, same_raw, label="identical target/draft p=q output")
