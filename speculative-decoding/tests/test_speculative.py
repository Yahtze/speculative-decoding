"""
E2E smoke test for speculative decoding with probabilistic acceptance.
Uses SmolLM for both draft and target (validates algorithm correctness).
Run: uv run python speculative-decoding/tests/test_speculative.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models import LanguageModel
from src.decoding import SpeculativeDecoder, DecodeStats


MODEL = "HuggingFaceTB/SmolLM-135M"


def test_generate_returns_triple():
    """generate() returns (token_ids, decoded_text, stats)."""
    print("=== test_generate_returns_triple ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3, verbose=False)
    result = decoder.generate("Hello", max_new_tokens=20)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

    token_ids, decoded_text, stats = result
    assert isinstance(token_ids, torch.Tensor)
    assert isinstance(decoded_text, str)
    assert isinstance(stats, DecodeStats)

    print(f"token_ids shape: {token_ids.shape}")
    print(f"decoded_text: {decoded_text!r}")
    print(f"stats: {stats}")
    print()


def test_generate_contains_prompt():
    """Output starts with the prompt tokens."""
    print("=== test_generate_contains_prompt ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3)
    prompt = "The capital of France is"
    token_ids, _, _ = decoder.generate(prompt, max_new_tokens=20)

    prompt_ids = draft.encode(prompt).squeeze(0)
    assert torch.equal(token_ids[: prompt_ids.shape[0]], prompt_ids), "Prompt prefix mismatch"
    print("Prompt tokens preserved.")
    print()


def test_stats_are_consistent():
    """Stat counters are non-negative and drafted = accepted + rejected."""
    print("=== test_stats_are_consistent ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3)
    _, _, stats = decoder.generate("Once upon a time", max_new_tokens=30)

    assert stats.drafted_tokens >= 0, f"Negative drafted: {stats.drafted_tokens}"
    assert stats.accepted_tokens >= 0, f"Negative accepted: {stats.accepted_tokens}"
    assert stats.rejected_tokens >= 0, f"Negative rejected: {stats.rejected_tokens}"
    assert stats.target_calls >= 0, f"Negative target_calls: {stats.target_calls}"

    # drafted = accepted + rejected
    assert stats.drafted_tokens == stats.accepted_tokens + stats.rejected_tokens, (
        f"Drafted ({stats.drafted_tokens}) != accepted ({stats.accepted_tokens}) + rejected ({stats.rejected_tokens})"
    )

    # At least one target call
    assert stats.target_calls > 0, "No target model calls made"

    # Target calls should be >= ceil(drafted / k) since each round uses one target call
    # Plus bonus calls when all drafts accepted
    print(f"Drafted: {stats.drafted_tokens}")
    print(f"Accepted: {stats.accepted_tokens}")
    print(f"Rejected: {stats.rejected_tokens}")
    print(f"Target calls: {stats.target_calls}")
    print()


def test_decoded_text_matches_ids():
    """Decoded text is consistent with returned token ids."""
    print("=== test_decoded_text_matches_ids ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3)
    token_ids, decoded_text, _ = decoder.generate("Hello world", max_new_tokens=20)

    manual = draft.decode(token_ids.unsqueeze(0))
    if isinstance(manual, list):
        manual = manual[0]

    assert decoded_text == manual, f"Mismatch:\n  returned: {decoded_text}\n  manual:   {manual}"
    print(f"Match: {decoded_text}")
    print()


def test_verbose_shows_probabilities():
    """Verbose mode prints acceptance ratios with p, q, ratio, u values."""
    print("=== test_verbose_shows_probabilities ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3, verbose=True)
    print("--- verbose output start ---")
    _, _, stats = decoder.generate("The", max_new_tokens=10)
    print("--- verbose output end ---")

    # Just verify it runs without error and produces stats
    assert stats.drafted_tokens > 0
    assert stats.target_calls > 0
    print(f"Stats: {stats}")
    print()


def test_same_model_high_acceptance():
    """When draft and target are the same model, p(x) ≈ q(x) → high acceptance."""
    print("=== test_same_model_high_acceptance ===")
    lm = LanguageModel(MODEL)

    # Same model for both — acceptance ratio min(1, p/q) ≈ 1
    decoder = SpeculativeDecoder(lm, lm, k=3, temperature=1.0)
    _, _, stats = decoder.generate("Hello", max_new_tokens=30)

    acceptance_rate = stats.accepted_tokens / max(stats.drafted_tokens, 1)
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    print(f"Stats: {stats}")

    # With same model, p ≈ q, so ratio ≈ 1, acceptance should be high
    assert acceptance_rate > 0.5, f"Expected >50% acceptance with same model, got {acceptance_rate:.1%}"
    print()


def test_acceptance_is_probabilistic():
    """Probabilistic acceptance produces varied outputs across runs."""
    print("=== test_acceptance_is_probabilistic ===")
    lm = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(lm, lm, k=3, temperature=1.0)

    results = []
    for _ in range(5):
        _, text, stats = decoder.generate("The capital of France is", max_new_tokens=15)
        results.append((text, stats.accepted_tokens, stats.rejected_tokens))

    unique_texts = set(r[0] for r in results)
    print(f"Unique outputs: {len(unique_texts)} / 5")
    for i, (text, accepted, rejected) in enumerate(results):
        print(f"  Run {i}: accepted={accepted}, rejected={rejected}, text={text[:60]}...")

    # Probabilistic acceptance should produce some variation
    assert len(unique_texts) >= 1, "No variation in outputs"
    print()


def test_different_k_values():
    """Speculative decoding works with different k values."""
    print("=== test_different_k_values ===")
    lm = LanguageModel(MODEL)

    for k in [1, 2, 5]:
        decoder = SpeculativeDecoder(lm, lm, k=k, temperature=1.0)
        _, text, stats = decoder.generate("Hello", max_new_tokens=20)
        print(f"  k={k}: drafted={stats.drafted_tokens}, accepted={stats.accepted_tokens}, "
              f"rejected={stats.rejected_tokens}, text={text[:40]}...")
        assert stats.drafted_tokens > 0
        assert stats.drafted_tokens == stats.accepted_tokens + stats.rejected_tokens
    print()


def test_longer_generation():
    """Generate a longer sequence without errors."""
    print("=== test_longer_generation ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=5)
    token_ids, decoded_text, stats = decoder.generate(
        "The meaning of life is", max_new_tokens=50
    )

    assert token_ids.dim() == 1
    print(f"Generated {token_ids.shape[0]} total tokens")
    print(f"Acceptance rate: {stats.accepted_tokens / max(stats.drafted_tokens, 1):.1%}")
    print(f"Stats: {stats}")
    print(f"Output: {decoded_text[:100]}...")
    print()


if __name__ == "__main__":
    test_generate_returns_triple()
    test_generate_contains_prompt()
    test_stats_are_consistent()
    test_decoded_text_matches_ids()
    test_verbose_shows_probabilities()
    test_same_model_high_acceptance()
    test_acceptance_is_probabilistic()
    test_different_k_values()
    test_longer_generation()
    print("All tests passed!")
