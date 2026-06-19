"""
Wrappers that abstract away Hugging Face internals.
Provides clean interfaces for language models and embedding models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .loader import load_model, load_tokenizer


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class LMOutput:
    """Container for language model forward pass results."""

    logits: torch.Tensor  # (batch, seq_len, vocab_size)
    hidden_states: Optional[torch.Tensor] = None  # (batch, seq_len, hidden_dim)
    loss: Optional[torch.Tensor] = None


@dataclass
class EmbeddingOutput:
    """Container for embedding model forward pass results."""

    embeddings: torch.Tensor  # (batch, seq_len, hidden_dim) or (batch, hidden_dim)


# ---------------------------------------------------------------------------
# Language Model Wrapper
# ---------------------------------------------------------------------------


class LanguageModel:
    """
    Thin wrapper around Hugging Face causal language models.

    Hides tokenizer/model internals and exposes:
      - forward / backward passes
      - shape properties (batch_size, vocab_size, sequence_length, hidden_size)
      - device / dtype introspection
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | str | None = None,
        **kwargs,
    ):
        self._tokenizer: AutoTokenizer = load_tokenizer(model_name)
        self._model: AutoModelForCausalLM = load_model(
            model_name, device=device, dtype=dtype, **kwargs
        )
        self._last_input_ids: Optional[torch.Tensor] = None

    # -- properties ---------------------------------------------------------

    @property
    def model(self) -> AutoModelForCausalLM:
        """Underlying Hugging Face model."""
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Underlying Hugging Face tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Device the model parameters live on."""
        return next(self._model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the model parameters."""
        return next(self._model.parameters()).dtype

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (number of tokens)."""
        return self._model.config.vocab_size

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the model."""
        return self._model.config.hidden_size

    @property
    def batch_size(self) -> Optional[int]:
        """Batch size of the last forward pass, or None if not yet called."""
        if self._last_input_ids is None:
            return None
        return self._last_input_ids.shape[0]

    @property
    def sequence_length(self) -> Optional[int]:
        """Sequence length of the last forward pass, or None if not yet called."""
        if self._last_input_ids is None:
            return None
        return self._last_input_ids.shape[1]

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self._model.parameters())

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self._model.config.num_hidden_layers

    # -- forward / backward -------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> LMOutput:
        """
        Run a forward pass.

        Args:
            input_ids: Token ids (batch, seq_len).
            attention_mask: Mask for padding tokens (batch, seq_len).
            labels: Target ids for loss computation (batch, seq_len).
            return_hidden_states: If True, include last hidden state in output.

        Returns:
            LMOutput with logits, optional loss, optional hidden_states.
        """
        self._last_input_ids = input_ids

        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )

        hidden_states = None
        if return_hidden_states and output.hidden_states is not None:
            hidden_states = output.hidden_states[-1]

        return LMOutput(
            logits=output.logits,
            hidden_states=hidden_states,
            loss=output.loss,
        )

    def backward(self, loss: torch.Tensor) -> None:
        """
        Run backward pass on a loss tensor.

        Args:
            loss: Scalar loss to backpropagate.
        """
        loss.backward()

    # -- convenience --------------------------------------------------------

    def encode(self, text: str | list[str]) -> torch.Tensor:
        """Tokenize text and return input_ids on the model's device."""
        return self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str | list[str]:
        """Decode token ids back to text."""
        return self._tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        params = f"{self.num_parameters / 1e6:.1f}M"
        return (
            f"LanguageModel(vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, params={params}, device='{self.device}', dtype={self.dtype})"
        )


# ---------------------------------------------------------------------------
# Embedding Model Wrapper
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """
    Thin wrapper around Hugging Face embedding / encoder models.

    Hides tokenizer/model internals and exposes:
      - forward pass (returns embeddings)
      - shape properties (vocab_size, hidden_size, max_position_embeddings)
      - device / dtype introspection
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | str | None = None,
        **kwargs,
    ):
        self._tokenizer: AutoTokenizer = load_tokenizer(model_name)
        self._model: AutoModel = AutoModel.from_pretrained(model_name, **kwargs)
        self._model.eval()

        # Move to device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        if dtype is not None:
            self._model = self._model.to(dtype=dtype)
        self._model = self._model.to(self._device)

        self._last_input_ids: Optional[torch.Tensor] = None

    # -- properties ---------------------------------------------------------

    @property
    def model(self) -> AutoModel:
        """Underlying Hugging Face model."""
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Underlying Hugging Face tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        """Device the model parameters live on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the model parameters."""
        return next(self._model.parameters()).dtype

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (number of tokens)."""
        return self._model.config.vocab_size

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the model."""
        return self._model.config.hidden_size

    @property
    def max_position_embeddings(self) -> int:
        """Maximum sequence length the model supports."""
        return self._model.config.max_position_embeddings

    @property
    def batch_size(self) -> Optional[int]:
        """Batch size of the last forward pass, or None if not yet called."""
        if self._last_input_ids is None:
            return None
        return self._last_input_ids.shape[0]

    @property
    def sequence_length(self) -> Optional[int]:
        """Sequence length of the last forward pass, or None if not yet called."""
        if self._last_input_ids is None:
            return None
        return self._last_input_ids.shape[1]

    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self._model.parameters())

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "cls",
    ) -> EmbeddingOutput:
        """
        Run a forward pass and return embeddings.

        Args:
            input_ids: Token ids (batch, seq_len).
            attention_mask: Mask for padding tokens (batch, seq_len).
            pooling: How to reduce token embeddings to a single vector:
                     "cls" — use the [CLS] token embedding.
                     "mean" — mean-pool over non-padding tokens.
                     "none" — return all token embeddings (batch, seq_len, hidden).

        Returns:
            EmbeddingOutput with embeddings tensor.
        """
        self._last_input_ids = input_ids

        output = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden = output.last_hidden_state  # (batch, seq_len, hidden)

        if pooling == "cls":
            embeddings = last_hidden[:, 0, :]  # (batch, hidden)
        elif pooling == "mean":
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        elif pooling == "none":
            embeddings = last_hidden
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling!r}")

        return EmbeddingOutput(embeddings=embeddings)

    # -- convenience --------------------------------------------------------

    def encode(self, text: str | list[str]) -> torch.Tensor:
        """Tokenize text and return input_ids on the model's device."""
        return self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(self.device)

    def embed(self, text: str | list[str], pooling: str = "cls") -> torch.Tensor:
        """
        Convenience: tokenize + forward in one call.

        Args:
            text: Input string or list of strings.
            pooling: Pooling strategy ("cls", "mean", "none").

        Returns:
            Embeddings tensor.
        """
        input_ids = self.encode(text)
        return self.forward(input_ids, pooling=pooling).embeddings

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        params = f"{self.num_parameters / 1e6:.1f}M"
        return (
            f"EmbeddingModel(vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"max_seq_len={self.max_position_embeddings}, params={params}, "
            f"device='{self.device}', dtype={self.dtype})"
        )
