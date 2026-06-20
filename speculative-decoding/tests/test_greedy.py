"""
E2E tests for greedy decoding using SmolLM.
Run: uv run python speculative-decoding/tests/test_greedy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models import LanguageModel
from src.decoding import greedy_decode


MODEL = "HuggingFaceTB/SmolLM-135M"


def test_greedy_returns_tuple():
    """greedy_decode returns (token_ids, decoded_text)."""
    print("=== test_greedy_returns_tuple ===")
    lm = LanguageModel(MODEL)
    result = greedy_decode(lm, "Hello", max_new_tokens=5)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    token_ids, decoded_text = result
    assert isinstance(token_ids, torch.Tensor), f"Expected Tensor, got {type(token_ids)}"
    assert isinstance(decoded_text, str), f"Expected str, got {type(decoded_text)}"
    print(f"token_ids: {token_ids}")
    print(f"decoded_text: {decoded_text}")
    print()


def test_greedy_token_ids_shape():
    """token_ids is 1-D and length >= prompt length."""
    print("=== test_greedy_token_ids_shape ===")
    lm = LanguageModel(MODEL)
    prompt = "The sky is"
    n = 10
    token_ids, _ = greedy_decode(lm, prompt, max_new_tokens=n)

    prompt_ids = lm.encode(prompt)
    prompt_len = prompt_ids.shape[1]

    assert token_ids.dim() == 1, f"Expected 1-D, got {token_ids.dim()}-D"
    assert token_ids.shape[0] >= prompt_len, (
        f"Expected >= {prompt_len}, got {token_ids.shape[0]}"
    )
    assert token_ids.shape[0] <= prompt_len + n, (
        f"Expected <= {prompt_len + n}, got {token_ids.shape[0]} (early stop possible)"
    )
    print(f"prompt_len: {prompt_len}, total_len: {token_ids.shape[0]}, max_new: {n}")
    print()


def test_greedy_contains_prompt():
    """Generated sequence starts with the prompt tokens."""
    print("=== test_greedy_contains_prompt ===")
    lm = LanguageModel(MODEL)
    prompt = "Once upon a time"
    token_ids, _ = greedy_decode(lm, prompt, max_new_tokens=5)

    prompt_ids = lm.encode(prompt).squeeze(0)
    generated_prefix = token_ids[: prompt_ids.shape[0]]

    assert torch.equal(generated_prefix, prompt_ids), (
        f"Prefix mismatch:\n  expected: {prompt_ids}\n  got:      {generated_prefix}"
    )
    print("Prompt tokens preserved at start of output.")
    print()


def test_greedy_deterministic():
    """Same prompt + same max_new_tokens = same output every time."""
    print("=== test_greedy_deterministic ===")
    lm = LanguageModel(MODEL)
    prompt = "2 + 2 ="
    n = 8

    ids1, text1 = greedy_decode(lm, prompt, max_new_tokens=n)
    ids2, text2 = greedy_decode(lm, prompt, max_new_tokens=n)

    assert torch.equal(ids1, ids2), "Token ids differ between runs"
    assert text1 == text2, f"Decoded text differs:\n  run 1: {text1}\n  run 2: {text2}"
    print(f"Deterministic output: {text1}")
    print()


def test_greedy_decoded_text_matches_ids():
    """Decoded text is consistent with the returned token ids."""
    print("=== test_greedy_decoded_text_matches_ids ===")
    lm = LanguageModel(MODEL)
    token_ids, decoded_text = greedy_decode(lm, "Hello world", max_new_tokens=5)

    manual_decode = lm.decode(token_ids.unsqueeze(0))
    if isinstance(manual_decode, list):
        manual_decode = manual_decode[0]

    assert decoded_text == manual_decode, (
        f"Mismatch:\n  returned: {decoded_text}\n  manual:   {manual_decode}"
    )
    print(f"decoded_text matches manual decode: {decoded_text}")
    print()


def test_greedy_max_new_tokens_zero():
    """max_new_tokens=0 returns the prompt unchanged."""
    print("=== test_greedy_max_new_tokens_zero ===")
    lm = LanguageModel(MODEL)
    prompt = "I am"
    token_ids, decoded_text = greedy_decode(lm, prompt, max_new_tokens=0)

    prompt_ids = lm.encode(prompt).squeeze(0)
    assert torch.equal(token_ids, prompt_ids), "Expected prompt-only output"
    print(f"Zero generation: {decoded_text}")
    print()


def test_greedy_long_generation():
    """Generate a longer sequence without errors."""
    print("=== test_greedy_long_generation ===")
    lm = LanguageModel(MODEL)
    prompt = "The meaning of life is"
    n = 50
    token_ids, decoded_text = greedy_decode(lm, prompt, max_new_tokens=n)

    assert token_ids.dim() == 1
    print(f"Generated {token_ids.shape[0]} tokens")
    print(f"Output: {decoded_text}")
    print()


if __name__ == "__main__":
    test_greedy_returns_tuple()
    test_greedy_token_ids_shape()
    test_greedy_contains_prompt()
    test_greedy_deterministic()
    test_greedy_decoded_text_matches_ids()
    test_greedy_max_new_tokens_zero()
    test_greedy_long_generation()
    print("All tests passed!")
