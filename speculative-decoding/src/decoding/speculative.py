"""
Speculative decoding: draft-then-verify with top-k sampling.

A small draft model proposes K tokens cheaply.
A large target model verifies them in one forward pass.
At the first mismatch, we keep the target's token and discard the rest.
Repeats until EOS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..models.wrapper import LanguageModel


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class DecodeStats:
    """Counters for a single speculative-decoding run."""

    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    target_calls: int = 0


# ---------------------------------------------------------------------------
# Speculative Decoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """
    Speculative decoding with top-k sampling.

    Args:
        draft_model: Small, fast model that proposes candidate tokens.
        target_model: Large, accurate model that verifies candidates.
        k: Number of tokens the draft model proposes per round.
        temperature: Sampling temperature for both models.
        top_k: Number of highest-probability tokens to sample from.
        verbose: If True, print detailed step-by-step logs.
    """

    def __init__(
        self,
        draft_model: LanguageModel,
        target_model: LanguageModel,
        k: int = 3,
        temperature: float = 1.0,
        top_k: int = 50,
        verbose: bool = False,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = k
        self.temperature = temperature
        self.top_k = top_k
        self.verbose = verbose

    # -- helpers ------------------------------------------------------------

    def _top_k_sample(self, model: LanguageModel, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass and sample from the top-k distribution.

        Returns:
            next_token_id: (1, 1) tensor with the sampled token.
        """
        output = model.forward(input_ids)
        logits = output.logits[:, -1, :]  # (1, vocab_size)

        # Mask to top-k
        top_k_vals, _ = torch.topk(logits, self.top_k, dim=-1)
        threshold = top_k_vals[:, -1].unsqueeze(-1)
        logits[logits < threshold] = float("-inf")

        # Sample
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        return next_token

    def _decode_token(self, model: LanguageModel, token_id: int) -> str:
        """Decode a single token id to its string representation."""
        return model.tokenizer.decode([token_id], skip_special_tokens=True)

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.verbose:
            print(message)

    # -- core steps ---------------------------------------------------------

    @torch.no_grad()
    def draft_step(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draft phase: run the small model autoregressively for k tokens.

        Args:
            input_ids: Current sequence (1, seq_len).
            eos_token_id: EOS token id to stop early.

        Returns:
            (drafted_ids, extended_input_ids)
            - drafted_ids: (1, k') tensor of newly generated tokens (k' <= k if EOS hit).
            - extended_input_ids: (1, seq_len + k') full sequence with drafts appended.
        """
        drafted_tokens: list[torch.Tensor] = []
        current_ids = input_ids.clone()

        for _ in range(self.k):
            next_token = self._top_k_sample(self.draft_model, current_ids)
            drafted_tokens.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        drafted_ids = torch.cat(drafted_tokens, dim=-1)  # (1, num_drafted)
        extended_input_ids = torch.cat([input_ids, drafted_ids], dim=-1)

        # Log draft tokens
        draft_texts = [self._decode_token(self.draft_model, t.item()) for t in drafted_tokens]
        self._log(f"DRAFT:    {''.join(draft_texts)!r}")

        return drafted_ids, extended_input_ids

    @torch.no_grad()
    def verify_step(
        self,
        extended_input_ids: torch.Tensor,
        num_drafted: int,
    ) -> torch.Tensor:
        """
        Verify phase: run the target model on the full extended sequence
        in a single forward pass and extract target tokens for the drafted positions.

        Args:
            extended_input_ids: (1, seq_len + num_drafted) full sequence.
            num_drafted: How many tokens were drafted.

        Returns:
            target_token_ids: (1, num_drafted) tensor — the target model's preferred
                token at each drafted position.
        """
        output = self.target_model.forward(extended_input_ids)
        logits = output.logits  # (1, seq_len + num_drafted, vocab_size)

        # For each drafted position, the target model's prediction is the
        # argmax of the logits at the *previous* position.
        # Drafted tokens occupy positions [original_len .. original_len + num_drafted - 1].
        # The target model's prediction for position i comes from logits at position i-1.
        original_len = extended_input_ids.shape[1] - num_drafted

        target_token_ids = []
        for i in range(num_drafted):
            pos = original_len + i - 1  # logits at pos predict token at pos+1
            pos_logits = logits[:, pos, :]  # (1, vocab_size)

            # Greedy: take argmax (deterministic verification)
            target_token = pos_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            target_token_ids.append(target_token)

        target_ids = torch.cat(target_token_ids, dim=-1)  # (1, num_drafted)

        # Log target tokens
        target_texts = [self._decode_token(self.target_model, t.item()) for t in target_token_ids]
        self._log(f"TARGET:   {''.join(target_texts)!r}")

        return target_ids

    @torch.no_grad()
    def acceptance_step(
        self,
        input_ids: torch.Tensor,
        drafted_ids: torch.Tensor,
        target_ids: torch.Tensor,
        eos_token_id: int | None,
    ) -> tuple[torch.Tensor, bool, int, int]:
        """
        Compare draft and target tokens. Accept matches, reject at first mismatch.

        At a mismatch, we keep the target model's token for that position
        and discard all remaining draft tokens.

        Args:
            input_ids: Original sequence before drafting (1, seq_len).
            drafted_ids: (1, num_drafted) tokens from the draft model.
            target_ids: (1, num_drafted) tokens from the target model.
            eos_token_id: EOS token id to detect end of generation.

        Returns:
            (new_input_ids, hit_eos, num_accepted, num_rejected)
            - new_input_ids: Updated sequence after accepting/rejecting.
            - hit_eos: True if an accepted token was EOS.
            - num_accepted: Number of tokens accepted this round.
            - num_rejected: Number of tokens rejected this round.
        """
        num_drafted = drafted_ids.shape[1]
        accepted_tokens: list[torch.Tensor] = []
        hit_eos = False

        for i in range(num_drafted):
            draft_tok = drafted_ids[0, i].item()
            target_tok = target_ids[0, i].item()

            if draft_tok == target_tok:
                # Match — accept the draft token
                accepted_tokens.append(drafted_ids[0, i : i + 1])
                self._log(
                    f"ACCEPTED: {self._decode_token(self.draft_model, draft_tok)!r}"
                )
                if eos_token_id is not None and draft_tok == eos_token_id:
                    hit_eos = True
                    break
            else:
                # Mismatch — use the target's token, discard the rest
                accepted_tokens.append(target_ids[0, i : i + 1])
                self._log(
                    f"REJECTED: {self._decode_token(self.draft_model, draft_tok)!r}"
                )
                self._log(
                    f"  (used target token: {self._decode_token(self.target_model, target_tok)!r})"
                )
                if eos_token_id is not None and target_tok == eos_token_id:
                    hit_eos = True
                break

        num_accepted = len(accepted_tokens)
        num_rejected = num_drafted - num_accepted

        if accepted_tokens:
            accepted_ids = torch.cat(accepted_tokens, dim=-1).unsqueeze(0)  # (1, n)
            new_input_ids = torch.cat([input_ids, accepted_ids], dim=-1)
        else:
            # Edge case: mismatch at position 0, but we always accept target's token
            new_input_ids = input_ids

        return new_input_ids, hit_eos, num_accepted, num_rejected

    # -- main loop ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> tuple[torch.Tensor, str, DecodeStats]:
        """
        Run speculative decoding until EOS or max_new_tokens total tokens generated.

        Args:
            prompt: Text prompt.
            max_new_tokens: Hard cap on total generated tokens.

        Returns:
            (token_ids, decoded_text, stats)
        """
        stats = DecodeStats()
        eos_token_id = self.draft_model.tokenizer.eos_token_id

        # Encode prompt (use draft model's tokenizer — both should share vocab)
        input_ids = self.draft_model.encode(prompt)  # (1, seq_len)
        original_len = input_ids.shape[1]

        self._log(f"PROMPT:   {prompt!r}")
        self._log("-" * 40)

        total_generated = 0

        while total_generated < max_new_tokens:
            # 1. Draft
            drafted_ids, extended_input_ids = self.draft_step(input_ids, eos_token_id)
            num_drafted = drafted_ids.shape[1]
            stats.drafted_tokens += num_drafted

            # 2. Verify (one target call per round)
            target_ids = self.verify_step(extended_input_ids, num_drafted)
            stats.target_calls += 1

            # 3. Accept / reject
            input_ids, hit_eos, num_accepted, num_rejected = self.acceptance_step(
                input_ids, drafted_ids, target_ids, eos_token_id
            )
            stats.accepted_tokens += num_accepted
            stats.rejected_tokens += num_rejected
            total_generated = input_ids.shape[1] - original_len

            self._log("-" * 40)

            if hit_eos:
                break

        # Decode final sequence
        decoded_text: str = self.draft_model.decode(input_ids)
        if isinstance(decoded_text, list):
            decoded_text = decoded_text[0]

        return input_ids.squeeze(0), decoded_text, stats
