"""
E2E smoke test for speculative decoding using SmolLM.
Uses the same small model for both draft and target (just to verify the algorithm).
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


def test_stats_are_positive():
    """All stat counters should be non-negative and drafted >= accepted."""
    print("=== test_stats_are_positive ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3)
    _, _, stats = decoder.generate("Once upon a time", max_new_tokens=30)

    assert stats.drafted_tokens >= 0
    assert stats.accepted_tokens >= 0
    assert stats.rejected_tokens >= 0
    assert stats.target_calls >= 0

    # drafted = accepted + rejected (per round, but sum should match)
    assert stats.drafted_tokens == stats.accepted_tokens + stats.rejected_tokens, (
        f"Drafted ({stats.drafted_tokens}) != accepted ({stats.accepted_tokens}) + rejected ({stats.rejected_tokens})"
    )

    # At least one target call happened
    assert stats.target_calls > 0

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


def test_verbose_output():
    """Verbose mode should print draft/target/accepted/rejected logs."""
    print("=== test_verbose_output ===")
    draft = LanguageModel(MODEL)
    target = LanguageModel(MODEL)

    decoder = SpeculativeDecoder(draft, target, k=3, verbose=True)
    print("--- verbose output start ---")
    _, _, stats = decoder.generate("The", max_new_tokens=10)
    print("--- verbose output end ---")
    print(f"Stats: {stats}")
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
    print(f"Stats: {stats}")
    print(f"Output: {decoded_text}")
    print()


def test_same_model_accepts_all():
    """When draft and target are the same model, most tokens should be accepted."""
    print("=== test_same_model_accepts_all ===")
    lm = LanguageModel(MODEL)

    # Same model for both — should have high acceptance rate
    decoder = SpeculativeDecoder(lm, lm, k=3, temperature=0.01)
    _, _, stats = decoder.generate("Hello", max_new_tokens=30)

    acceptance_rate = stats.accepted_tokens / max(stats.drafted_tokens, 1)
    print(f"Acceptance rate: {acceptance_rate:.1%}")
    print(f"Stats: {stats}")

    # With same model and low temperature, acceptance should be high
    assert acceptance_rate > 0.5, f"Expected high acceptance with same model, got {acceptance_rate:.1%}"
    print()


if __name__ == "__main__":
    test_generate_returns_triple()
    test_generate_contains_prompt()
    test_stats_are_positive()
    test_decoded_text_matches_ids()
    test_verbose_output()
    test_longer_generation()
    test_same_model_accepts_all()
    print("All tests passed!")
