"""
Sampling-based decoding: temperature scaling + softmax + random sampling.

Three strategies:
  - sample_decode: full-vocab sampling with temperature
  - top_k_decode: sample only from the k highest-probability tokens
  - top_p_decode: sample from the smallest set of tokens whose cumulative
    probability >= p (nucleus sampling)
"""

from __future__ import annotations

import torch

from ..models.wrapper import LanguageModel


def _sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by temperature, apply softmax, and sample one token.

    Args:
        logits: Raw logits (batch=1, vocab_size).
        temperature: Scaling factor. Lower = sharper (more greedy),
                     higher = flatter (more random). Must be > 0.

    Returns:
        Sampled token id (1, 1).
    """
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
    return next_token


@torch.no_grad()
def sample_decode(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, str]:
    """
    Generate tokens by sampling from the temperature-scaled distribution.

    Args:
        model: LanguageModel wrapper.
        prompt: Text prompt to start generation from.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature. 1.0 = unchanged logits,
                     <1.0 = sharper (more greedy), >1.0 = flatter (more random).

    Returns:
        (token_ids, decoded_text)
        - token_ids: 1-D tensor of the full sequence (prompt + generated).
        - decoded_text: Decoded string of the full sequence.
    """
    assert temperature > 0, f"Temperature must be > 0, got {temperature}"

    input_ids: torch.Tensor = model.encode(prompt)  # (1, seq_len)
    eos_token_id = model.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        output = model.forward(input_ids)
        logits: torch.Tensor = output.logits
        last_token_logits: torch.Tensor = logits[:, -1, :]  # (1, vocab_size)

        next_token_id = _sample_from_logits(last_token_logits, temperature)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    decoded_text: str = model.decode(input_ids)
    if isinstance(decoded_text, list):
        decoded_text = decoded_text[0]

    return input_ids.squeeze(0), decoded_text


@torch.no_grad()
def top_k_decode(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    k: int = 50,
) -> tuple[torch.Tensor, str]:
    """
    Generate tokens by sampling from the top-k highest-logit tokens.

    At each step:
      1. Get logits for the last position.
      2. Find the k-th largest logit value.
      3. Set all logits below that threshold to -inf.
      4. Apply temperature scaling + softmax + sample.

    Args:
        model: LanguageModel wrapper.
        prompt: Text prompt to start generation from.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        k: Number of top tokens to keep. Must be >= 1.

    Returns:
        (token_ids, decoded_text)
        - token_ids: 1-D tensor of the full sequence (prompt + generated).
        - decoded_text: Decoded string of the full sequence.
    """
    assert temperature > 0, f"Temperature must be > 0, got {temperature}"
    assert k >= 1, f"k must be >= 1, got {k}"

    input_ids: torch.Tensor = model.encode(prompt)  # (1, seq_len)
    eos_token_id = model.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        output = model.forward(input_ids)
        logits: torch.Tensor = output.logits
        last_token_logits: torch.Tensor = logits[:, -1, :]  # (1, vocab_size)

        # Find the k-th largest value — everything below this becomes -inf
        top_k_vals, _ = torch.topk(last_token_logits, k, dim=-1)  # (1, k)
        threshold = top_k_vals[:, -1].unsqueeze(-1)  # (1, 1) — smallest of top-k
        mask = last_token_logits < threshold
        last_token_logits[mask] = float("-inf")

        next_token_id = _sample_from_logits(last_token_logits, temperature)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    decoded_text: str = model.decode(input_ids)
    if isinstance(decoded_text, list):
        decoded_text = decoded_text[0]

    return input_ids.squeeze(0), decoded_text


@torch.no_grad()
def top_p_decode(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    p: float = 0.9,
) -> tuple[torch.Tensor, str]:
    """
    Generate tokens by nucleus (top-p) sampling.

    At each step:
      1. Scale logits by temperature and apply softmax.
      2. Sort probabilities in descending order.
      3. Compute cumulative sum of sorted probabilities.
      4. Find the first token whose cumulative sum exceeds p.
      5. Zero out all tokens at and beyond that point.
      6. Renormalize the remaining probabilities.
      7. Sample from the truncated distribution.

    Args:
        model: LanguageModel wrapper.
        prompt: Text prompt to start generation from.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        p: Cumulative probability threshold. Keeps the smallest set of
           tokens whose total probability >= p. Must be in (0, 1].
           p=1.0 keeps all tokens (equivalent to full-vocab sampling).

    Returns:
        (token_ids, decoded_text)
        - token_ids: 1-D tensor of the full sequence (prompt + generated).
        - decoded_text: Decoded string of the full sequence.
    """
    assert temperature > 0, f"Temperature must be > 0, got {temperature}"
    assert 0 < p <= 1, f"p must be in (0, 1], got {p}"

    input_ids: torch.Tensor = model.encode(prompt)  # (1, seq_len)
    eos_token_id = model.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        output = model.forward(input_ids)
        logits: torch.Tensor = output.logits
        last_token_logits: torch.Tensor = logits[:, -1, :]  # (1, vocab_size)

        # Scale and convert to probabilities
        scaled = last_token_logits / temperature
        probs = torch.softmax(scaled, dim=-1)  # (1, vocab_size)

        # Sort descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Cumulative sum of sorted probabilities
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Shift cumsum right by 1 so we include the token that crosses p,
        # not the one after it.  cumshift[i] = cumsum[i-1] (with cumshift[0]=0).
        cumshift = torch.zeros_like(cumsum)
        cumshift[:, 1:] = cumsum[:, :-1]

        # Mask: keep tokens whose cumulative probability *before* them is < p
        # This means we keep the smallest set whose total >= p.
        mask = cumshift < p

        # Zero out tokens outside the nucleus, then renormalize
        sorted_probs[~mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample from the truncated distribution (indices into sorted order)
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)

        # Map back to original vocab index
        next_token_id = sorted_indices.gather(-1, sampled_sorted_idx)

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    decoded_text: str = model.decode(input_ids)
    if isinstance(decoded_text, list):
        decoded_text = decoded_text[0]

    return input_ids.squeeze(0), decoded_text
