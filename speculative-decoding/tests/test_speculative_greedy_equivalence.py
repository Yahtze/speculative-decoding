"""Token-for-token oracles for greedy speculative decoding (no downloads)."""

import pytest
import torch
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from src.decoding import SpeculativeDecoder, greedy_decode
from src.models import LanguageModel

from .decoding_fixtures import deterministic_cycle


@pytest.mark.parametrize("draft_offset", [1, 2], ids=["matching", "rejecting"])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("limit", [1, 2, 5, 8])
def test_greedy_matches_target_only(draft_offset, k, limit):
    # Different next-token tables make a wrong verification index observable.
    draft = deterministic_cycle(draft_offset)
    target = deterministic_cycle()
    expected_ids, expected_text = greedy_decode(target, "prompt", limit)
    actual_ids, actual_text, _ = SpeculativeDecoder(
        draft, target, k=k, temperature=0, top_k=4
    ).generate("prompt", limit)
    assert actual_ids.tolist() == expected_ids.tolist(), (
        f"k={k}, limit={limit}, draft_offset={draft_offset}: "
        f"speculative={actual_ids.tolist()}, target_only={expected_ids.tolist()}"
    )
    assert actual_text == expected_text


@pytest.fixture(scope="module")
def local_hf_models(tmp_path_factory):
    """Exercise save/load, a real tokenizer, real transformer and real wrappers."""
    root = tmp_path_factory.mktemp("local-hf")
    vocab = {
        "[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "Hello": 3,
        "world": 4, "café": 5, "你好": 6, "compiler": 7,
    }
    backend = BackendTokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
        eos_token="[EOS]", pad_token="[EOS]", model_max_length=64,
    )
    models = []
    for name, seed in [("draft", 491), ("target", 947)]:
        path = root / name
        torch.manual_seed(seed)
        config = GPT2Config(
            vocab_size=len(vocab), n_positions=64, n_ctx=64,
            n_embd=16, n_layer=1, n_head=2, bos_token_id=1,
            # Disable automatic EOS to test exact output length with random weights.
            eos_token_id=None, pad_token_id=2,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        )
        model = GPT2LMHeadModel(config).eval()
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        models.append(LanguageModel(str(path), device="cpu", dtype=torch.float32))
    # Use no EOS in the tokenizer too; EOS-specific cases have their own suite.
    for model in models:
        model.tokenizer.eos_token = None
    return models


@pytest.mark.parametrize("prompt", ["Hello world", "café 你好"])
@pytest.mark.parametrize("limit", [1, 7])
@pytest.mark.parametrize("same_model", [True, False], ids=["shared-model", "distinct-models"])
@torch.no_grad()
def test_hf_end_to_end_greedy_matches_generate(local_hf_models, prompt, limit, same_model):
    draft, target = local_hf_models
    if same_model:
        draft = target
    encoded = target.encode(prompt)
    # Independent Hugging Face generation oracle, not the project greedy helper.
    expected = target.model.generate(
        encoded, attention_mask=torch.ones_like(encoded), do_sample=False,
        max_new_tokens=limit, eos_token_id=None, pad_token_id=2,
        use_cache=False,
    ).squeeze(0)
    baseline, _ = greedy_decode(target, prompt, max_new_tokens=limit)
    assert torch.equal(baseline, expected), "Target-only helper differs from HF generate"
    actual, text, _ = SpeculativeDecoder(
        draft, target, k=3, temperature=0, top_k=8,
    ).generate(prompt, max_new_tokens=limit)
    print(
        f"HF greedy: prompt={prompt!r}, limit={limit}, same_model={same_model}, "
        f"expected={expected.tolist()}, actual={actual.tolist()}"
    )
    assert torch.equal(actual, expected)
    assert text == target.tokenizer.decode(actual, skip_special_tokens=True)
