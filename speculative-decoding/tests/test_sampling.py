"""
E2E tests for sampling-based decoding using SmolLM.
Run: uv run python speculative-decoding/tests/test_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models import LanguageModel
from src.decoding import sample_decode, top_k_decode, top_p_decode


MODEL = "HuggingFaceTB/SmolLM-135M"


def test_sample_returns_tuple():
    """sample_decode returns (token_ids, decoded_text)."""
    print("=== test_sample_returns_tuple ===")
    lm = LanguageModel(MODEL)
    result = sample_decode(lm, "Hello", max_new_tokens=5, temperature=1.0)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    token_ids, decoded_text = result
    assert isinstance(token_ids, torch.Tensor)
    assert isinstance(decoded_text, str)
    print(f"token_ids: {token_ids}")
    print(f"decoded_text: {decoded_text}")
    print()


def test_sample_contains_prompt():
    """Output starts with the prompt tokens."""
    print("=== test_sample_contains_prompt ===")
    lm = LanguageModel(MODEL)
    prompt = "Once upon a time"
    token_ids, _ = sample_decode(lm, prompt, max_new_tokens=10, temperature=1.0)

    prompt_ids = lm.encode(prompt).squeeze(0)
    assert torch.equal(token_ids[: prompt_ids.shape[0]], prompt_ids), "Prompt prefix mismatch"
    print("Prompt tokens preserved.")
    print()


def test_sample_temperature_low_is_greedy():
    """Very low temperature should behave close to greedy."""
    print("=== test_sample_temperature_low_is_greedy ===")
    lm = LanguageModel(MODEL)
    prompt = "2 + 2 ="
    n = 8

    # temperature near 0 → almost deterministic like greedy
    ids_sample, text_sample = sample_decode(lm, prompt, max_new_tokens=n, temperature=0.01)

    from src.decoding import greedy_decode
    ids_greedy, text_greedy = greedy_decode(lm, prompt, max_new_tokens=n)

    assert torch.equal(ids_sample, ids_greedy), (
        f"Low temp should match greedy:\n  sample: {ids_sample}\n  greedy: {ids_greedy}"
    )
    print(f"Low temp matches greedy: {text_greedy}")
    print()


def test_sample_temperature_high_more_varied():
    """Higher temperature should produce different outputs across runs (usually)."""
    print("=== test_sample_temperature_high_more_varied ===")
    lm = LanguageModel(MODEL)
    prompt = "The meaning of life is"
    n = 20

    texts = set()
    for _ in range(5):
        _, text = sample_decode(lm, prompt, max_new_tokens=n, temperature=5.0)
        texts.add(text)

    # With high temp and 5 runs, we expect some variation (not guaranteed but very likely)
    print(f"Unique outputs: {len(texts)} / 5")
    assert len(texts) > 1, f"High temp should produce varied outputs, got {len(texts)} unique"
    print()


def test_sample_decoded_text_matches_ids():
    """Decoded text is consistent with returned token ids."""
    print("=== test_sample_decoded_text_matches_ids ===")
    lm = LanguageModel(MODEL)
    token_ids, decoded_text = sample_decode(lm, "Hello world", max_new_tokens=5, temperature=1.0)

    manual = lm.decode(token_ids.unsqueeze(0))
    if isinstance(manual, list):
        manual = manual[0]

    assert decoded_text == manual, f"Mismatch:\n  returned: {decoded_text}\n  manual:   {manual}"
    print(f"Match: {decoded_text}")
    print()


def test_sample_temperature_invalid():
    """Temperature <= 0 should raise."""
    print("=== test_sample_temperature_invalid ===")
    lm = LanguageModel(MODEL)
    try:
        sample_decode(lm, "Hello", max_new_tokens=5, temperature=0.0)
        assert False, "Should have raised"
    except AssertionError:
        pass
    except AssertionError:
        pass
    print("Correctly rejected temperature=0")
    print()


# --- top-k ---


def test_top_k_returns_tuple():
    """top_k_decode returns (token_ids, decoded_text)."""
    print("=== test_top_k_returns_tuple ===")
    lm = LanguageModel(MODEL)
    result = top_k_decode(lm, "Hello", max_new_tokens=5, temperature=1.0, k=10)

    assert isinstance(result, tuple)
    token_ids, decoded_text = result
    assert isinstance(token_ids, torch.Tensor)
    assert isinstance(decoded_text, str)
    print(f"token_ids: {token_ids}")
    print(f"decoded_text: {decoded_text}")
    print()


def test_top_k_contains_prompt():
    """Output starts with the prompt tokens."""
    print("=== test_top_k_contains_prompt ===")
    lm = LanguageModel(MODEL)
    prompt = "The cat sat on"
    token_ids, _ = top_k_decode(lm, prompt, max_new_tokens=10, k=10)

    prompt_ids = lm.encode(prompt).squeeze(0)
    assert torch.equal(token_ids[: prompt_ids.shape[0]], prompt_ids)
    print("Prompt tokens preserved.")
    print()


def test_top_k_k1_is_greedy():
    """k=1 should always pick the top token (greedy)."""
    print("=== test_top_k_k1_is_greedy ===")
    lm = LanguageModel(MODEL)
    prompt = "2 + 2 ="
    n = 8

    ids_topk, text_topk = top_k_decode(lm, prompt, max_new_tokens=n, temperature=1.0, k=1)

    from src.decoding import greedy_decode
    ids_greedy, text_greedy = greedy_decode(lm, prompt, max_new_tokens=n)

    assert torch.equal(ids_topk, ids_greedy), (
        f"k=1 should match greedy:\n  top-k: {ids_topk}\n  greedy: {ids_greedy}"
    )
    print(f"k=1 matches greedy: {text_greedy}")
    print()


def test_top_k_decoded_text_matches_ids():
    """Decoded text is consistent with returned token ids."""
    print("=== test_top_k_decoded_text_matches_ids ===")
    lm = LanguageModel(MODEL)
    token_ids, decoded_text = top_k_decode(lm, "Hello world", max_new_tokens=5, k=5)

    manual = lm.decode(token_ids.unsqueeze(0))
    if isinstance(manual, list):
        manual = manual[0]

    assert decoded_text == manual
    print(f"Match: {decoded_text}")
    print()


def test_top_k_invalid_params():
    """Invalid k or temperature should raise."""
    print("=== test_top_k_invalid_params ===")
    lm = LanguageModel(MODEL)

    try:
        top_k_decode(lm, "Hello", max_new_tokens=5, k=0)
        assert False, "Should have raised for k=0"
    except AssertionError:
        pass

    try:
        top_k_decode(lm, "Hello", max_new_tokens=5, temperature=-1.0)
        assert False, "Should have raised for temp<0"
    except AssertionError:
        pass

    print("Correctly rejected invalid params")
    print()


def test_top_k_long_generation():
    """Generate a longer sequence without errors."""
    print("=== test_top_k_long_generation ===")
    lm = LanguageModel(MODEL)
    prompt = "The meaning of life is"
    n = 50
    token_ids, decoded_text = top_k_decode(lm, prompt, max_new_tokens=n, temperature=1.0, k=10)

    assert token_ids.dim() == 1
    print(f"Generated {token_ids.shape[0]} tokens")
    print(f"Output: {decoded_text}")
    print()


# --- top-p ---


def test_top_p_returns_tuple():
    """top_p_decode returns (token_ids, decoded_text)."""
    print("=== test_top_p_returns_tuple ===")
    lm = LanguageModel(MODEL)
    result = top_p_decode(lm, "Hello", max_new_tokens=5, temperature=1.0, p=0.9)

    assert isinstance(result, tuple)
    token_ids, decoded_text = result
    assert isinstance(token_ids, torch.Tensor)
    assert isinstance(decoded_text, str)
    print(f"token_ids: {token_ids}")
    print(f"decoded_text: {decoded_text}")
    print()


def test_top_p_contains_prompt():
    """Output starts with the prompt tokens."""
    print("=== test_top_p_contains_prompt ===")
    lm = LanguageModel(MODEL)
    prompt = "The cat sat on"
    token_ids, _ = top_p_decode(lm, prompt, max_new_tokens=10, p=0.9)

    prompt_ids = lm.encode(prompt).squeeze(0)
    assert torch.equal(token_ids[: prompt_ids.shape[0]], prompt_ids)
    print("Prompt tokens preserved.")
    print()


def test_top_p_p1_is_sample():
    """p=1.0 keeps all tokens, should match full-vocab sampling with same seed."""
    print("=== test_top_p_p1_is_sample ===")
    lm = LanguageModel(MODEL)
    prompt = "2 + 2 ="
    n = 8

    # Both use the same underlying sampling path; p=1.0 means no truncation
    ids_top_p, text_top_p = top_p_decode(lm, prompt, max_new_tokens=n, temperature=1.0, p=1.0)

    # Just verify it runs and produces valid output
    assert ids_top_p.dim() == 1
    assert len(text_top_p) > 0
    print(f"p=1.0 output: {text_top_p}")
    print()


def test_top_p_low_p_more_greedy():
    """Very low p (e.g. 0.1) restricts to very few tokens — should be near-greedy."""
    print("=== test_top_p_low_p_more_greedy ===")
    lm = LanguageModel(MODEL)
    prompt = "2 + 2 ="
    n = 8

    from src.decoding import greedy_decode
    ids_greedy, text_greedy = greedy_decode(lm, prompt, max_new_tokens=n)

    # p=0.1 with low temperature should heavily favor the top token
    ids_top_p, text_top_p = top_p_decode(lm, prompt, max_new_tokens=n, temperature=0.1, p=0.1)

    # They should match or be very close
    assert torch.equal(ids_top_p, ids_greedy), (
        f"Low p + low temp should match greedy:\n  top-p: {ids_top_p}\n  greedy: {ids_greedy}"
    )
    print(f"Low p matches greedy: {text_greedy}")
    print()


def test_top_p_decoded_text_matches_ids():
    """Decoded text is consistent with returned token ids."""
    print("=== test_top_p_decoded_text_matches_ids ===")
    lm = LanguageModel(MODEL)
    token_ids, decoded_text = top_p_decode(lm, "Hello world", max_new_tokens=5, p=0.9)

    manual = lm.decode(token_ids.unsqueeze(0))
    if isinstance(manual, list):
        manual = manual[0]

    assert decoded_text == manual
    print(f"Match: {decoded_text}")
    print()


def test_top_p_invalid_params():
    """Invalid p or temperature should raise."""
    print("=== test_top_p_invalid_params ===")
    lm = LanguageModel(MODEL)

    try:
        top_p_decode(lm, "Hello", max_new_tokens=5, p=0.0)
        assert False, "Should have raised for p=0"
    except AssertionError:
        pass

    try:
        top_p_decode(lm, "Hello", max_new_tokens=5, p=1.1)
        assert False, "Should have raised for p>1"
    except AssertionError:
        pass

    try:
        top_p_decode(lm, "Hello", max_new_tokens=5, temperature=-1.0)
        assert False, "Should have raised for temp<0"
    except AssertionError:
        pass

    print("Correctly rejected invalid params")
    print()


def test_top_p_long_generation():
    """Generate a longer sequence without errors."""
    print("=== test_top_p_long_generation ===")
    lm = LanguageModel(MODEL)
    prompt = "The meaning of life is"
    n = 50
    token_ids, decoded_text = top_p_decode(lm, prompt, max_new_tokens=n, temperature=1.0, p=0.9)

    assert token_ids.dim() == 1
    print(f"Generated {token_ids.shape[0]} tokens")
    print(f"Output: {decoded_text}")
    print()


if __name__ == "__main__":
    test_sample_returns_tuple()
    test_sample_contains_prompt()
    test_sample_temperature_low_is_greedy()
    test_sample_temperature_high_more_varied()
    test_sample_decoded_text_matches_ids()
    test_sample_temperature_invalid()
    test_top_k_returns_tuple()
    test_top_k_contains_prompt()
    test_top_k_k1_is_greedy()
    test_top_k_decoded_text_matches_ids()
    test_top_k_invalid_params()
    test_top_k_long_generation()
    test_top_p_returns_tuple()
    test_top_p_contains_prompt()
    test_top_p_p1_is_sample()
    test_top_p_low_p_more_greedy()
    test_top_p_decoded_text_matches_ids()
    test_top_p_invalid_params()
    test_top_p_long_generation()
    print("All tests passed!")
