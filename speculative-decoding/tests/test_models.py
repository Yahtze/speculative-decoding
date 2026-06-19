"""
E2E tests for model wrappers using SmolLM.
Run: uv run python speculative-decoding/tests/test_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import LanguageModel, EmbeddingModel, Tokenizer


MODEL = "HuggingFaceTB/SmolLM-135M"


def test_tokenizer():
    print("=== Tokenizer ===")
    print(f"Model: {MODEL}")
    tok = Tokenizer(MODEL)
    print(tok)

    enc = tok.encode("Hello world")
    print(f"input_ids shape: {enc.input_ids.shape}")
    print(f"attention_mask shape: {enc.attention_mask.shape}")

    decoded = tok.decode(enc.input_ids)
    print(f"decoded: {decoded}")
    print()


def test_language_model():
    print("=== LanguageModel ===")
    print(f"Model: {MODEL}")
    lm = LanguageModel(MODEL)
    print(lm)

    ids = lm.encode("The quick brown fox")
    print(f"input_ids shape: {ids.shape}")

    out = lm.forward(ids, labels=ids)
    print(f"logits shape: {out.logits.shape}")
    print(f"loss: {out.loss.item():.4f}")
    print(f"batch_size: {lm.batch_size}")
    print(f"sequence_length: {lm.sequence_length}")
    print(f"vocab_size: {lm.vocab_size}")
    print(f"hidden_size: {lm.hidden_size}")
    print(f"num_layers: {lm.num_layers}")
    print(f"num_parameters: {lm.num_parameters / 1e6:.1f}M")
    print()


def test_embedding_model():
    print("=== EmbeddingModel ===")
    print(f"Model: {MODEL}")
    emb = EmbeddingModel(MODEL)
    print(emb)

    ids = emb.encode("Hello world")
    print(f"input_ids shape: {ids.shape}")

    out = emb.forward(ids, pooling="cls")
    print(f"embeddings shape (cls): {out.embeddings.shape}")

    out = emb.forward(ids, pooling="mean")
    print(f"embeddings shape (mean): {out.embeddings.shape}")

    vec = emb.embed("Hello world", pooling="mean")
    print(f"embed() shape: {vec.shape}")
    print()


if __name__ == "__main__":
    test_tokenizer()
    test_language_model()
    test_embedding_model()
    print("All tests passed!")
