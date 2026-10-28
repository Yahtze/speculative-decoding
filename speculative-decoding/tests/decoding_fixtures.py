"""Small causal models with independently specified next-token distributions.

These doubles exercise the real decoder; they do not implement its algorithm.
The optional transition matrix makes verification-position errors observable.
"""

from types import SimpleNamespace

import torch


class ToyTokenizer:
    def __init__(self, vocab_size, eos_token_id=None, vocabulary=None):
        self.eos_token_id = eos_token_id
        self.bos_token_id = 0
        self.pad_token_id = eos_token_id
        self._vocabulary = vocabulary or {f"t{i}": i for i in range(vocab_size)}

    def get_vocab(self):
        return dict(self._vocabulary)

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(
            f"t{int(token)}"
            for token in token_ids
            if not skip_special_tokens or int(token) != self.eos_token_id
        )


class ToyLanguageModel:
    def __init__(
        self, probabilities, *, eos_token_id=None, encoded_ids=None, tokenizer=None
    ):
        self.probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
        self.vocab_size = self.probabilities.shape[-1]
        self.device = torch.device("cpu")
        self.tokenizer = tokenizer or ToyTokenizer(self.vocab_size, eos_token_id)
        self.encoded_ids = [0] if encoded_ids is None else list(encoded_ids)
        self.forward_calls = 0
        self.inputs = []

    def forward(self, input_ids):
        self.forward_calls += 1
        # Retain just the last call: statistical tests make many calls.
        self.inputs = [input_ids.clone()]
        if self.probabilities.ndim == 1:
            probs = self.probabilities.expand(*input_ids.shape, -1)
        else:
            probs = self.probabilities[input_ids]
        return SimpleNamespace(logits=probs.log().clone())

    def encode(self, prompt):
        return torch.tensor([self.encoded_ids], dtype=torch.long)

    def decode(self, input_ids):
        return [self.tokenizer.decode(row.tolist()) for row in input_ids]

    def __repr__(self):
        return f"ToyLanguageModel(vocab_size={self.vocab_size})"


def deterministic_cycle(offset=1, vocab_size=4, **kwargs):
    table = torch.zeros((vocab_size, vocab_size), dtype=torch.float64)
    for token in range(vocab_size):
        table[token, (token + offset) % vocab_size] = 1
    return ToyLanguageModel(table, **kwargs)
