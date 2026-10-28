"""Boundary tests for EOS, tokenization, length limits and reported counters."""

import json

import pytest
import torch

from src.benchmarks import run_experiment
from src.decoding import SpeculativeDecoder, greedy_decode

from .decoding_fixtures import ToyLanguageModel, deterministic_cycle


def decode(draft, target=None, *, k=3, limit=8, **kwargs):
    return SpeculativeDecoder(
        draft, target or draft, k=k, temperature=0,
        top_k=draft.vocab_size, **kwargs,
    ).generate("prompt", max_new_tokens=limit)


def test_accepted_draft_eos_stops_without_bonus():
    model = ToyLanguageModel([0, 0, 1], eos_token_id=2)
    ids, _, stats = decode(model)
    assert ids.tolist() == [0, 2]
    assert stats.accepted_tokens == 1
    assert stats.target_calls == 1


def test_target_eos_on_rejection_stops_immediately():
    draft = ToyLanguageModel([0, 1, 0], eos_token_id=2)
    target = ToyLanguageModel([0, 0, 1], eos_token_id=2)
    ids, _, _ = decode(draft, target)
    assert ids.tolist() == [0, 2]


def test_rejected_draft_eos_does_not_stop_generation():
    draft = ToyLanguageModel([0, 0, 1], eos_token_id=2)
    target = ToyLanguageModel(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]], eos_token_id=2
    )
    expected, _ = greedy_decode(target, "prompt", 8)
    ids, _, _ = decode(draft, target)
    assert ids.tolist() == expected.tolist() == [0, 1, 2]


def test_bonus_eos_stops_without_another_round():
    model = ToyLanguageModel(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]], eos_token_id=2
    )
    ids, _, stats = decode(model, k=1)
    assert ids.tolist() == [0, 1, 2]
    assert stats.draft_calls == 1
    assert stats.target_calls == 2


def test_eos_in_prompt_does_not_end_new_generation():
    model = deterministic_cycle(encoded_ids=[2], eos_token_id=2)
    ids, _, _ = decode(model)
    assert ids.tolist() == [2, 3, 0, 1, 2]


def test_missing_eos_runs_to_limit():
    model = deterministic_cycle()
    ids, _, _ = decode(model, k=2, limit=6)
    assert ids.numel() == 7


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 5])
def test_max_new_tokens_is_hard_cap(limit):
    model = deterministic_cycle()
    ids, _, _ = decode(model, k=3, limit=limit)
    assert ids.numel() == 1 + limit, (
        f"max_new_tokens={limit} generated {ids.numel() - 1} tokens"
    )


def test_zero_budget_makes_no_forward_calls():
    model = deterministic_cycle(encoded_ids=[0, 1])
    ids, _, stats = decode(model, limit=0)
    assert ids.tolist() == [0, 1]
    assert model.forward_calls == stats.target_calls == stats.draft_calls == 0


def test_prompt_token_ids_preserved():
    model = deterministic_cycle(encoded_ids=[3, 1, 0])
    ids, text, _ = decode(model, limit=3)
    assert ids[:3].tolist() == [3, 1, 0]
    assert text == model.tokenizer.decode(ids.tolist())


def test_zero_acceptance_ratio_never_accepts(monkeypatch):
    # torch.rand samples [0, 1), so an exact zero is a valid RNG outcome.
    draft = ToyLanguageModel([0, 1, 0])
    target = ToyLanguageModel([0, 0, 1])
    decoder = SpeculativeDecoder(draft, target, k=1, temperature=0, top_k=3)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(1))
    ids, _, _, _ = decoder.acceptance_step(
        torch.tensor([[0]]), torch.tensor([[1]]),
        torch.tensor([1.0]), torch.tensor([0.0]), None,
    )
    assert ids.shape == (1, 2), "acceptance_step must preserve the batch dimension"
    assert ids[0].tolist() == [0, 2], "A token with target probability zero was accepted"


def test_replacement_does_not_count_as_accepted_draft():
    draft = ToyLanguageModel([0, 1, 0])
    target = ToyLanguageModel([0, 0, 1])
    _, _, stats = decode(draft, target, k=3, limit=1)
    assert stats.accepted_tokens == 0
    assert stats.rejected_tokens == stats.drafted_tokens


def test_model_call_counters_measure_forward_calls():
    draft = ToyLanguageModel([0, 1, 0])
    target = ToyLanguageModel([0, 0, 1])
    _, _, stats = decode(draft, target, k=3, limit=1)
    assert stats.draft_calls == draft.forward_calls
    assert stats.target_calls == target.forward_calls


def test_throughput_includes_bonus_tokens():
    model = deterministic_cycle()
    ids, _, stats = decode(model, k=1, limit=2)
    assert stats.tokens_per_second == pytest.approx(
        (ids.numel() - 1) / stats.generation_time
    )


def test_batch_saved_token_total_matches_outputs(tmp_path):
    draft, target = deterministic_cycle(), deterministic_cycle()
    result = run_experiment(
        draft, target, k=1, prompts=["first", "second"],
        max_new_tokens=2, temperature=0, top_k=4,
        save_name="edge-batch", results_dir=tmp_path,
    )
    assert len(result.prompt_results) == 2
    data = json.loads((tmp_path / "edge-batch.json").read_text())
    # Every output contains the one-token prompt and two generated tokens.
    generated = sum(len(item.output.split()) - 1 for item in result.prompt_results)
    assert data["totals"]["total_tokens_generated"] == generated == 4
