"""Real tokenizer and transformer edge cases, built locally without downloads."""

import pytest
import torch

from src.decoding import SpeculativeDecoder

# Reuse the local, saved-and-reloaded real transformer/tokenizer fixture.
from .test_speculative_greedy_equivalence import local_hf_models


@pytest.mark.parametrize("prompt", ["Hello world", "café 你好", "[BOS] Hello world"])
def test_real_tokenization_zero_budget_preserves_ids(local_hf_models, prompt):
    _, model = local_hf_models
    expected = model.encode(prompt).squeeze(0)
    ids, text, stats = SpeculativeDecoder(
        model, model, k=3, temperature=0, top_k=8
    ).generate(prompt, max_new_tokens=0)
    assert torch.equal(ids, expected)
    assert text == model.tokenizer.decode(expected, skip_special_tokens=True)
    assert stats.drafted_tokens == stats.target_calls == 0


@pytest.mark.parametrize("prompt", ["Hello world", "café 你好"])
def test_real_unicode_round_trip(local_hf_models, prompt):
    _, model = local_hf_models
    ids = model.encode(prompt)
    assert 0 not in ids.tolist()[0], "Unexpected unknown token"
    assert model.decode(ids) == [prompt]


def test_padding_fallback_to_eos(tmp_path):
    from src.models.loader import load_tokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "Hello": 2}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(tmp_path)
    actual = load_tokenizer(str(tmp_path))
    assert actual.pad_token_id == actual.eos_token_id == 1


@torch.no_grad()
def test_real_eos_included_once_and_matches_target_generate(local_hf_models):
    _, model = local_hf_models
    encoded = model.encode("Hello world")
    # Select a reproducible EOS that this real transformer will emit next.
    eos = model.forward(encoded).logits[:, -1].argmax().item()
    token = model.tokenizer.convert_ids_to_tokens(eos)
    # The backend exposes special tokens dynamically. Restore with assignment,
    # not monkeypatch.delattr, which its token-property API does not support.
    model.tokenizer.eos_token = token
    try:
        expected = model.model.generate(
            encoded, attention_mask=torch.ones_like(encoded), max_new_tokens=8,
            do_sample=False, use_cache=False, eos_token_id=eos, pad_token_id=2,
        ).squeeze(0)
        ids, text, stats = SpeculativeDecoder(
            model, model, k=3, temperature=0, top_k=8
        ).generate("Hello world", max_new_tokens=8)
        assert torch.equal(ids, expected)
        assert ids.numel() == encoded.numel() + 1
        assert ids[-1].item() == eos
        assert stats.target_calls == 1
        assert text == model.tokenizer.decode(ids, skip_special_tokens=True)
    finally:
        model.tokenizer.eos_token = None
