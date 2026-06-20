"""
Greedy decoding: at each step pick the token with highest probability.
"""

from __future__ import annotations

import torch

from ..models.wrapper import LanguageModel


@torch.no_grad()
def greedy_decode(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int = 50,
) -> tuple[torch.Tensor, str]:
    """
    Generate tokens greedily by repeatedly taking argmax of the last-logit
    distribution and appending the chosen token id to the sequence.

    Args:
        model: LanguageModel wrapper (has .forward, .encode, .decode, .tokenizer).
        prompt: Text prompt to start generation from.
        max_new_tokens: Number of new tokens to generate.

    Returns:
        (token_ids, decoded_text)
        - token_ids: 1-D tensor of the full sequence (prompt + generated).
        - decoded_text: Decoded string of the full sequence (special tokens stripped).
    """
    input_ids: torch.Tensor = model.encode(prompt)  # (1, seq_len)

    eos_token_id = model.tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        # Forward pass -> logits (batch=1, seq_len, vocab_size)
        output = model.forward(input_ids)
        logits: torch.Tensor = output.logits

        # Take distribution over vocab at the last position
        last_token_logits: torch.Tensor = logits[:, -1, :]  # (1, vocab_size)

        # Greedy: pick token with highest logit
        next_token_id: torch.Tensor = last_token_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Stop early if EOS generated
        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    decoded_text: str = model.decode(input_ids)
    if isinstance(decoded_text, list):
        decoded_text = decoded_text[0]

    return input_ids.squeeze(0), decoded_text
