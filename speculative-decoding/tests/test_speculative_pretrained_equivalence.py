"""Compare the actual pretrained SmolLM wrapper with HF target-only generation.

This integration test downloads the same model used by the original smoke tests.
The offline correctness suites do not need this model.
"""

import pytest
import torch

from src.decoding import SpeculativeDecoder, greedy_decode
from src.models import LanguageModel


@pytest.fixture(scope="module")
def pretrained_target():
    torch.manual_seed(20261028)
    return LanguageModel("HuggingFaceTB/SmolLM-135M", device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("prompt", ["The capital of France is", "café 你好"])
@pytest.mark.parametrize("limit", [1, 7])
@torch.no_grad()
def test_pretrained_greedy_matches_hf_generate(pretrained_target, prompt, limit):
    model = pretrained_target
    ids = model.encode(prompt)
    expected = model.model.generate(
        ids, attention_mask=torch.ones_like(ids), do_sample=False,
        max_new_tokens=limit, use_cache=False,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    ).squeeze(0)
    baseline, _ = greedy_decode(model, prompt, max_new_tokens=limit)
    assert torch.equal(baseline, expected), "Target-only helper disagrees with HF generate"
    actual, text, _ = SpeculativeDecoder(
        model, model, k=3, temperature=0, top_k=50
    ).generate(prompt, max_new_tokens=limit)
    print(
        f"SmolLM greedy: prompt={prompt!r}, max_new_tokens={limit}, "
        f"expected={expected.tolist()}, actual={actual.tolist()}"
    )
    assert torch.equal(actual, expected)
    assert text == model.tokenizer.decode(expected, skip_special_tokens=True)
