"""
Speculative decoding: draft-then-verify with probabilistic acceptance.

A small draft model proposes K tokens cheaply via top-k sampling.
A large target model verifies them in one forward pass.
Each draft token is accepted with probability min(1, p(x)/q(x)),
where p(x) is the target probability and q(x) is the draft probability.
On rejection, we resample from the adjusted target distribution.
Repeats until EOS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..models.wrapper import LanguageModel


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class DecodeStats:
    """Counters and metrics for a single speculative-decoding run."""

    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    draft_calls: int = 0
    target_calls: int = 0
    tokenization_time: float = 0.0
    generation_time: float = 0.0
    decode_time: float = 0.0

    @property
    def total_time(self) -> float:
        """End-to-end time for one prompt."""
        return self.tokenization_time + self.generation_time + self.decode_time

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted (0.0 to 1.0)."""
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens

    @property
    def rejection_rate(self) -> float:
        """Fraction of drafted tokens that were rejected (0.0 to 1.0)."""
        if self.drafted_tokens == 0:
            return 0.0
        return self.rejected_tokens / self.drafted_tokens

    @property
    def tokens_per_second(self) -> float:
        """Tokens generated per second (including prompt)."""
        if self.generation_time == 0:
            return 0.0
        return (self.accepted_tokens + self.rejected_tokens) / self.generation_time


# ---------------------------------------------------------------------------
# Speculative Decoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """
    Speculative decoding with top-k sampling and probabilistic acceptance.

    Algorithm (per round):
      1. Draft model generates K tokens autoregressively via top-k sampling,
         recording the draft probability q(x) for each token.
      2. Target model runs one forward pass on the extended sequence,
         extracting target probability p(x) for each draft token.
      3. For each draft token, accept with probability min(1, p(x)/q(x)).
         On first rejection, resample a token from the adjusted distribution
         max(0, p(x) - q(x)) and discard remaining drafts.

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

    def _top_k_sample_with_probs(
        self, model: LanguageModel, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one forward pass, apply top-k masking, and sample.

        Returns:
            (next_token_id, token_prob)
            - next_token_id: (1, 1) sampled token.
            - token_prob: scalar tensor with the probability of the sampled token.
        """
        output = model.forward(input_ids)
        logits = output.logits[:, -1, :]  # (1, vocab_size)

        # Mask to top-k
        top_k_vals, _ = torch.topk(logits, self.top_k, dim=-1)
        threshold = top_k_vals[:, -1].unsqueeze(-1)
        mask = logits < threshold
        logits[mask] = float("-inf")

        # Greedy mode: argmax instead of sampling
        if self.temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            # One-hot probability for greedy token
            probs = torch.zeros_like(logits)
            probs[0, next_token.item()] = 1.0
            token_prob = probs[0, next_token.item()]
            return next_token, token_prob

        # Scale and convert to probabilities
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)  # (1, vocab_size)

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        token_prob = probs[0, next_token.item()]  # scalar

        return next_token, token_prob

    def _get_probs(
        self, model: LanguageModel, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the full probability distribution over the vocabulary for the last position.

        Returns:
            probs: (1, vocab_size) probability distribution.
        """
        output = model.forward(input_ids)
        logits = output.logits[:, -1, :]  # (1, vocab_size)

        # Greedy mode: one-hot on argmax
        if self.temperature == 0.0:
            probs = torch.zeros_like(logits)
            probs[0, logits.argmax(dim=-1)] = 1.0
            return probs

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)
        return probs

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Draft phase: run the small model autoregressively for k tokens.

        Args:
            input_ids: Current sequence (1, seq_len).
            eos_token_id: EOS token id to stop early.

        Returns:
            (drafted_ids, extended_input_ids, draft_probs)
            - drafted_ids: (1, k') tensor of newly generated tokens.
            - extended_input_ids: (1, seq_len + k') full sequence with drafts appended.
            - draft_probs: (k',) tensor of q(x) for each drafted token.
        """
        drafted_tokens: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] = []
        current_ids = input_ids.clone()

        for _ in range(self.k):
            next_token, token_prob = self._top_k_sample_with_probs(
                self.draft_model, current_ids
            )
            drafted_tokens.append(next_token)
            draft_probs.append(token_prob)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        drafted_ids = torch.cat(drafted_tokens, dim=-1)  # (1, num_drafted)
        draft_probs_tensor = torch.stack(draft_probs)  # (num_drafted,)
        extended_input_ids = torch.cat([input_ids, drafted_ids], dim=-1)

        # Log draft tokens
        draft_texts = [self._decode_token(self.draft_model, t.item()) for t in drafted_tokens]
        self._log(f"DRAFT:    {''.join(draft_texts)!r}")

        return drafted_ids, extended_input_ids, draft_probs_tensor

    @torch.no_grad()
    def verify_step(
        self,
        extended_input_ids: torch.Tensor,
        drafted_ids: torch.Tensor,
        num_drafted: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Verify phase: run the target model on the full extended sequence
        in a single forward pass and extract target probabilities for the draft tokens.

        Args:
            extended_input_ids: (1, seq_len + num_drafted) full sequence.
            drafted_ids: (1, num_drafted) tokens from the draft model.
            num_drafted: How many tokens were drafted.

        Returns:
            (target_probs, target_token_ids)
            - target_probs: (num_drafted,) tensor of p(x) for each draft token.
            - target_token_ids: (1, num_drafted) argmax tokens from target (for logging).
        """
        output = self.target_model.forward(extended_input_ids)
        logits = output.logits  # (1, seq_len + num_drafted, vocab_size)

        original_len = extended_input_ids.shape[1] - num_drafted

        target_probs: list[torch.Tensor] = []
        target_token_ids: list[torch.Tensor] = []

        for i in range(num_drafted):
            pos = original_len + i - 1  # logits at pos predict token at pos+1
            pos_logits = logits[:, pos, :]  # (1, vocab_size)

            draft_tok = drafted_ids[0, i].item()

            # Greedy mode: one-hot on argmax
            if self.temperature == 0.0:
                probs = torch.zeros_like(pos_logits)
                argmax_tok = pos_logits.argmax(dim=-1, keepdim=True)
                probs[0, argmax_tok.item()] = 1.0
                target_prob = probs[0, draft_tok]
                target_probs.append(target_prob)
                target_token_ids.append(argmax_tok)
                continue

            scaled = pos_logits / self.temperature
            probs = torch.softmax(scaled, dim=-1)  # (1, vocab_size)

            target_prob = probs[0, draft_tok]  # p(x) for the draft token
            target_probs.append(target_prob)

            # Also get argmax for logging
            argmax_tok = probs.argmax(dim=-1, keepdim=True)  # (1, 1)
            target_token_ids.append(argmax_tok)

        target_probs_tensor = torch.stack(target_probs)  # (num_drafted,)
        target_ids = torch.cat(target_token_ids, dim=-1)  # (1, num_drafted)

        # Log target tokens
        target_texts = [self._decode_token(self.target_model, t.item()) for t in target_token_ids]
        self._log(f"TARGET:   {''.join(target_texts)!r}")

        return target_probs_tensor, target_ids

    @torch.no_grad()
    def acceptance_step(
        self,
        input_ids: torch.Tensor,
        drafted_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        eos_token_id: int | None,
    ) -> tuple[torch.Tensor, bool, int, int]:
        """
        Probabilistic acceptance: accept each draft token with probability min(1, p/q).

        On rejection at position i:
          - Resample from the adjusted distribution: clamp(p(x) - q(x), 0, inf), renormalized.
          - If the resampled token is different, use it and discard remaining drafts.
          - If we accept all drafts, sample one bonus token from the target.

        Args:
            input_ids: Original sequence before drafting (1, seq_len).
            drafted_ids: (1, num_drafted) tokens from the draft model.
            draft_probs: (num_drafted,) q(x) probabilities from draft model.
            target_probs: (num_drafted,) p(x) probabilities from target model.
            eos_token_id: EOS token id to detect end of generation.

        Returns:
            (new_input_ids, hit_eos, num_accepted, num_rejected)
        """
        num_drafted = drafted_ids.shape[1]
        accepted_tokens: list[torch.Tensor] = []
        hit_eos = False

        for i in range(num_drafted):
            draft_tok = drafted_ids[0, i].item()
            p_x = target_probs[i].item()
            q_x = draft_probs[i].item()

            # Acceptance probability: min(1, p(x) / q(x))
            if q_x > 0:
                accept_ratio = min(1.0, p_x / q_x)
            else:
                accept_ratio = 1.0  # If draft prob is 0, always accept target's view

            u = torch.rand(1).item()

            if u <= accept_ratio:
                # Accept the draft token
                accepted_tokens.append(drafted_ids[0, i : i + 1])
                self._log(
                    f"ACCEPTED: {self._decode_token(self.draft_model, draft_tok)!r} "
                    f"(p={p_x:.4f}, q={q_x:.4f}, ratio={accept_ratio:.4f}, u={u:.4f})"
                )
                if eos_token_id is not None and draft_tok == eos_token_id:
                    hit_eos = True
                    break
            else:
                # Reject — resample from adjusted distribution max(0, p(x) - q(x))
                self._log(
                    f"REJECTED: {self._decode_token(self.draft_model, draft_tok)!r} "
                    f"(p={p_x:.4f}, q={q_x:.4f}, ratio={accept_ratio:.4f}, u={u:.4f})"
                )

                # Get full target distribution and resample
                resampled_token = self._resample_rejected(input_ids, accepted_tokens, eos_token_id)
                accepted_tokens.append(resampled_token)

                resampled_text = self._decode_token(self.target_model, resampled_token.item())
                self._log(f"  (resampled target token: {resampled_text!r})")

                if eos_token_id is not None and resampled_token.item() == eos_token_id:
                    hit_eos = True
                break

        num_accepted = len(accepted_tokens)
        num_rejected = num_drafted - num_accepted

        if accepted_tokens:
            accepted_ids = torch.cat(accepted_tokens, dim=-1).unsqueeze(0)  # (1, n)
            new_input_ids = torch.cat([input_ids, accepted_ids], dim=-1)
        else:
            new_input_ids = input_ids

        return new_input_ids, hit_eos, num_accepted, num_rejected

    @torch.no_grad()
    def _resample_rejected(
        self,
        input_ids: torch.Tensor,
        accepted_tokens: list[torch.Tensor],
        eos_token_id: int | None,
    ) -> torch.Tensor:
        """
        Resample from the adjusted distribution when a draft token is rejected.

        The adjusted distribution is: p'(x) = norm(max(0, p(x) - q(x)))
        where p(x) is the target distribution and q(x) is the draft distribution.

        If we have no accepted tokens yet, we just sample from the target distribution
        at the current position. Otherwise, we need to construct the sequence with
        accepted tokens appended and sample from there.
        """
        # Build the sequence up to the rejection point
        if accepted_tokens:
            accepted_ids = torch.cat(accepted_tokens, dim=-1).unsqueeze(0)  # (1, n)
            current_ids = torch.cat([input_ids, accepted_ids], dim=-1)
        else:
            current_ids = input_ids

        # Get target distribution at the current position
        target_probs = self._get_probs(self.target_model, current_ids)  # (1, vocab_size)

        # Get draft distribution at the current position
        draft_probs = self._get_probs(self.draft_model, current_ids)  # (1, vocab_size)

        # Adjusted distribution: max(0, p(x) - q(x))
        adjusted = torch.clamp(target_probs - draft_probs, min=0.0)  # (1, vocab_size)

        # Renormalize
        total = adjusted.sum(dim=-1, keepdim=True)
        if total > 0:
            adjusted = adjusted / total
        else:
            # Fallback: if adjusted is all zeros, use target distribution
            adjusted = target_probs

        # Sample from adjusted distribution
        next_token = torch.multinomial(adjusted, num_samples=1)  # (1, 1)
        return next_token.squeeze(0)  # (1,) to match draft token shape

    @torch.no_grad()
    def _sample_bonus_token(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """
        When all K draft tokens are accepted, sample one bonus token from the target.

        Returns:
            (bonus_token, hit_eos)
        """
        eos_token_id = self.draft_model.tokenizer.eos_token_id
        target_probs = self._get_probs(self.target_model, input_ids)  # (1, vocab_size)
        bonus_token = torch.multinomial(target_probs, num_samples=1)  # (1, 1)

        hit_eos = eos_token_id is not None and bonus_token.item() == eos_token_id
        return bonus_token, hit_eos

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
        tokenization_start = time.perf_counter()
        input_ids = self.draft_model.encode(prompt)  # (1, seq_len)
        stats.tokenization_time = time.perf_counter() - tokenization_start
        original_len = input_ids.shape[1]

        self._log(f"PROMPT:   {prompt!r}")
        self._log("-" * 40)

        total_generated = 0
        hit_eos = False
        start_time = time.perf_counter()

        while total_generated < max_new_tokens and not hit_eos:
            # 1. Draft
            drafted_ids, extended_input_ids, draft_probs = self.draft_step(
                input_ids, eos_token_id
            )
            num_drafted = drafted_ids.shape[1]
            stats.drafted_tokens += num_drafted
            stats.draft_calls += 1

            # 2. Verify (one target call per round)
            target_probs, target_ids = self.verify_step(
                extended_input_ids, drafted_ids, num_drafted
            )
            stats.target_calls += 1

            # 3. Accept / reject
            input_ids, hit_eos, num_accepted, num_rejected = self.acceptance_step(
                input_ids, drafted_ids, draft_probs, target_probs, eos_token_id
            )
            stats.accepted_tokens += num_accepted
            stats.rejected_tokens += num_rejected
            total_generated = input_ids.shape[1] - original_len

            # 4. If all K tokens accepted and no EOS, sample bonus token from target
            if num_accepted == num_drafted and not hit_eos and total_generated < max_new_tokens:
                bonus_token, bonus_eos = self._sample_bonus_token(input_ids)
                input_ids = torch.cat([input_ids, bonus_token], dim=-1)
                total_generated += 1
                stats.target_calls += 1

                bonus_text = self._decode_token(self.target_model, bonus_token.item())
                self._log(f"BONUS:    {bonus_text!r} (from target)")

                if bonus_eos:
                    hit_eos = True

            self._log("-" * 40)

        stats.generation_time = time.perf_counter() - start_time

        # Decode final sequence
        decode_start = time.perf_counter()
        decoded_text: str = self.draft_model.decode(input_ids)
        stats.decode_time = time.perf_counter() - decode_start
        if isinstance(decoded_text, list):
            decoded_text = decoded_text[0]

        return input_ids.squeeze(0), decoded_text, stats
