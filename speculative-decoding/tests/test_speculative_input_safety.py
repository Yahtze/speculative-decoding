"""Audit desired input guards, separate from current decoding guarantees.

The existing API assumes compatible tokenizers and useful model context. It
does not promise input validation or specific exception types. Failures here
record missing safety features, NOT target-distribution or greedy-law defects.
Keep this audit separate in the evidence summary to avoid false bug claims.
"""

import pytest

from src.decoding import SpeculativeDecoder

from .decoding_fixtures import ToyTokenizer, deterministic_cycle
from .test_speculative_edges import decode
from .test_speculative_greedy_equivalence import local_hf_models


def test_empty_prompt_fails_clearly_or_uses_bos():
    # Empty text is valid only if the decoder supplies an initial context token.
    model = deterministic_cycle(encoded_ids=[])
    try:
        ids, _, _ = decode(model, limit=1)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ["empty", "prompt", "context"])
    else:
        assert ids.numel() > 0, "Empty prompt must use BOS or raise a clear ValueError"


@pytest.mark.parametrize("mismatch", ["permuted_ids", "different_eos"])
def test_incompatible_tokenizers_are_rejected(mismatch):
    draft = deterministic_cycle()
    if mismatch == "permuted_ids":
        tokenizer = ToyTokenizer(
            4, vocabulary={"t0": 1, "t1": 0, "t2": 2, "t3": 3}
        )
    else:
        tokenizer = ToyTokenizer(4, eos_token_id=3)
    target = deterministic_cycle(tokenizer=tokenizer)
    with pytest.raises(ValueError, match="(?i)token|vocab|eos"):
        decode(draft, target)


def test_different_vocabulary_sizes_are_rejected():
    draft = deterministic_cycle(vocab_size=4)
    target = deterministic_cycle(vocab_size=3)
    with pytest.raises(ValueError, match="(?i)token|vocab"):
        decode(draft, target)


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"k": -1}, {"temperature": -1}, {"top_k": 0}]
)
def test_invalid_decoder_options_fail_clearly(kwargs):
    model = deterministic_cycle()
    options = dict(k=3, top_k=4, temperature=0)
    options.update(kwargs)
    with pytest.raises(ValueError):
        SpeculativeDecoder(model, model, **options).generate("prompt", 1)


def test_real_empty_prompt_has_safe_handling(local_hf_models):
    _, model = local_hf_models
    assert model.encode("").numel() == 0
    decoder = SpeculativeDecoder(model, model, k=3, temperature=0, top_k=8)
    try:
        ids, _, _ = decoder.generate("", max_new_tokens=1)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ["empty", "prompt", "context"])
    else:
        assert ids.numel() >= 1, "Supply BOS or raise a clear error for an empty prompt"
