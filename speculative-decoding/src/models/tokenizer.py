"""
Tokenizer wrapper for encoding and decoding text.
Uses the tokenizer loaded from loader.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer

from .loader import load_tokenizer


@dataclass
class Encoding:
    """Container for tokenization results."""

    input_ids: torch.Tensor  # (batch, seq_len)
    attention_mask: torch.Tensor  # (batch, seq_len)


class Tokenizer:
    """
    Thin wrapper around Hugging Face tokenizers.

    Provides clean encode/decode interface with:
      - batch encoding to tensors on specified device
      - decoding back to text
      - token id ↔ string conversions
      - vocabulary introspection
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        **kwargs,
    ):
        self._tokenizer: AutoTokenizer = load_tokenizer(model_name, **kwargs)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

    # -- properties ---------------------------------------------------------

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Underlying Hugging Face tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Device tensors are placed on."""
        return self._device

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> Optional[int]:
        """Pad token id."""
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """End of sequence token id."""
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Beginning of sequence token id."""
        return self._tokenizer.bos_token_id

    @property
    def model_max_length(self) -> int:
        """Maximum sequence length the tokenizer supports."""
        return self._tokenizer.model_max_length

    # -- encode / decode ----------------------------------------------------

    def encode(
        self,
        text: str | list[str],
        max_length: Optional[int] = None,
        padding: bool | str = True,
        truncation: bool = True,
    ) -> Encoding:
        """
        Encode text to token ids.

        Args:
            text: Input string or list of strings.
            max_length: Maximum sequence length (truncates if longer).
            padding: Pad to longest in batch (True), or no padding (False).
            truncation: Truncate to max_length if True.

        Returns:
            Encoding with input_ids and attention_mask tensors on self.device.
        """
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        return Encoding(
            input_ids=encoded.input_ids.to(self._device),
            attention_mask=encoded.attention_mask.to(self._device),
        )

    def decode(
        self,
        token_ids: torch.Tensor | list[int],
        skip_special_tokens: bool = True,
    ) -> str | list[str]:
        """
        Decode token ids back to text.

        Args:
            token_ids: Single sequence (1D) or batch (2D) of token ids.
            skip_special_tokens: Remove special tokens from output.

        Returns:
            Decoded string or list of strings.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        # Single sequence (1D)
        if token_ids and isinstance(token_ids[0], int):
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        # Batch (2D)
        return self._tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    # -- token ↔ string -----------------------------------------------------

    def token_to_id(self, token: str) -> int:
        """Convert a token string to its id."""
        return self._tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, token_id: int) -> str:
        """Convert a token id to its string representation."""
        return self._tokenizer.convert_ids_to_tokens(token_id)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, device='{self.device}', "
            f"max_length={self.model_max_length})"
        )
