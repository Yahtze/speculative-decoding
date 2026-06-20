"""
Batch benchmark runner for speculative decoding experiments.

Usage:
    from src.benchmarks.runner import run_experiment, ExperimentResult, load_prompts

    prompts = load_prompts("path/to/dataset")
    result = run_experiment(
        draft_model=LanguageModel("HuggingFaceTB/SmolLM-135M"),
        target_model=LanguageModel("HuggingFaceTB/SmolLM2-1.7B"),
        k=3,
        prompts=prompts,
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..models.wrapper import LanguageModel
from ..decoding.speculative import SpeculativeDecoder, DecodeStats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    """Result for a single prompt."""

    prompt: str
    output: str
    stats: DecodeStats


@dataclass
class ExperimentResult:
    """Aggregated results from a batch experiment."""

    # Core metrics (averaged across prompts)
    acceptance_rate: float = 0.0
    throughput: float = 0.0        # tokens per second
    latency: float = 0.0           # seconds per prompt
    target_calls: float = 0.0      # average target calls per prompt

    # Aggregate counters
    total_prompts: int = 0
    total_draft_calls: int = 0
    total_target_calls: int = 0
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    total_generation_time: float = 0.0
    total_tokens_generated: int = 0

    # Per-prompt results
    prompt_results: list[PromptResult] = field(default_factory=list)

    # Configuration
    draft_model_name: str = ""
    target_model_name: str = ""
    k: int = 0
    max_new_tokens: int = 0

    @property
    def rejection_rate(self) -> float:
        """Fraction of drafted tokens that were rejected."""
        if self.total_drafted_tokens == 0:
            return 0.0
        return self.total_rejected_tokens / self.total_drafted_tokens

    @property
    def tokens_per_target_call(self) -> float:
        """Average tokens generated per target model call."""
        if self.total_target_calls == 0:
            return 0.0
        return self.total_tokens_generated / self.total_target_calls

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Experiment Result\n"
            f"{'=' * 40}\n"
            f"Draft model:     {self.draft_model_name}\n"
            f"Target model:    {self.target_model_name}\n"
            f"k:               {self.k}\n"
            f"Max new tokens:  {self.max_new_tokens}\n"
            f"Prompts:         {self.total_prompts}\n"
            f"{'-' * 40}\n"
            f"Acceptance rate: {self.acceptance_rate:.1%}\n"
            f"Rejection rate:  {self.rejection_rate:.1%}\n"
            f"Throughput:      {self.throughput:.2f} tokens/s\n"
            f"Latency:         {self.latency:.3f} s/prompt\n"
            f"Target calls:    {self.target_calls:.1f} avg/prompt\n"
            f"Tokens/call:     {self.tokens_per_target_call:.2f}\n"
            f"{'-' * 40}\n"
            f"Total drafted:   {self.total_drafted_tokens}\n"
            f"Total accepted:  {self.total_accepted_tokens}\n"
            f"Total rejected:  {self.total_rejected_tokens}\n"
            f"Total draft calls:  {self.total_draft_calls}\n"
            f"Total target calls: {self.total_target_calls}\n"
            f"Total time:      {self.total_generation_time:.2f}s\n"
            f"Total tokens:    {self.total_tokens_generated}\n"
        )


# ---------------------------------------------------------------------------
# Prompt loading (placeholders — implement per dataset format)
# ---------------------------------------------------------------------------


def load_prompts(path: str | Path, format: str = "jsonl") -> list[str]:
    """
    Load prompts from a dataset file.

    Supported formats:
        - "jsonl": One JSON object per line with a "prompt" or "text" field
        - "txt": One prompt per line
        - "json": JSON array of strings

    Args:
        path: Path to the dataset file.
        format: Dataset format ("jsonl", "txt", or "json").

    Returns:
        List of prompt strings.
    """
    path = Path(path)

    if format == "jsonl":
        prompts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Try common field names
                if "prompt" in obj:
                    prompts.append(obj["prompt"])
                elif "text" in obj:
                    prompts.append(obj["text"])
                elif "input" in obj:
                    prompts.append(obj["input"])
                else:
                    raise ValueError(f"No prompt field found in: {obj.keys()}")
        return prompts

    elif format == "txt":
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    elif format == "json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(item) for item in data]
        raise ValueError("JSON format expects a list of strings")

    else:
        raise ValueError(f"Unsupported format: {format!r}")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    draft_model: LanguageModel,
    target_model: LanguageModel,
    k: int,
    prompts: list[str],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    verbose: bool = False,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> ExperimentResult:
    """
    Run speculative decoding across a batch of prompts.

    Args:
        draft_model: Small, fast model for draft proposals.
        target_model: Large, accurate model for verification.
        k: Number of tokens to draft per round.
        prompts: List of prompt strings to generate from.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-k filtering parameter.
        verbose: Print per-token logs for each prompt.
        progress_fn: Optional callback(current, total) for progress updates.

    Returns:
        ExperimentResult with aggregated metrics.
    """
    result = ExperimentResult(
        draft_model_name=repr(draft_model),
        target_model_name=repr(target_model),
        k=k,
        max_new_tokens=max_new_tokens,
        total_prompts=len(prompts),
    )

    decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        k=k,
        temperature=temperature,
        top_k=top_k,
        verbose=verbose,
    )

    for i, prompt in enumerate(prompts):
        # Run single prompt
        token_ids, output, stats = decoder.generate(prompt, max_new_tokens)

        # Record per-prompt result
        result.prompt_results.append(PromptResult(
            prompt=prompt,
            output=output,
            stats=stats,
        ))

        # Accumulate aggregates
        result.total_drafted_tokens += stats.drafted_tokens
        result.total_accepted_tokens += stats.accepted_tokens
        result.total_rejected_tokens += stats.rejected_tokens
        result.total_draft_calls += stats.draft_calls
        result.total_target_calls += stats.target_calls
        result.total_generation_time += stats.generation_time
        result.total_tokens_generated += (stats.accepted_tokens + stats.rejected_tokens)

        # Progress callback
        if progress_fn:
            progress_fn(i + 1, len(prompts))

    # Compute averages
    if result.total_prompts > 0:
        result.acceptance_rate = (
            result.total_accepted_tokens / max(result.total_drafted_tokens, 1)
        )
        result.throughput = (
            result.total_tokens_generated / max(result.total_generation_time, 0.001)
        )
        result.latency = result.total_generation_time / result.total_prompts
        result.target_calls = result.total_target_calls / result.total_prompts

    return result


# ---------------------------------------------------------------------------
# Convenience: save/load results
# ---------------------------------------------------------------------------


def save_result(result: ExperimentResult, path: str | Path) -> None:
    """Save experiment result to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "config": {
            "draft_model": result.draft_model_name,
            "target_model": result.target_model_name,
            "k": result.k,
            "max_new_tokens": result.max_new_tokens,
        },
        "metrics": {
            "acceptance_rate": result.acceptance_rate,
            "rejection_rate": result.rejection_rate,
            "throughput": result.throughput,
            "latency": result.latency,
            "target_calls": result.target_calls,
            "tokens_per_target_call": result.tokens_per_target_call,
        },
        "totals": {
            "total_prompts": result.total_prompts,
            "total_drafted_tokens": result.total_drafted_tokens,
            "total_accepted_tokens": result.total_accepted_tokens,
            "total_rejected_tokens": result.total_rejected_tokens,
            "total_draft_calls": result.total_draft_calls,
            "total_target_calls": result.total_target_calls,
            "total_generation_time": result.total_generation_time,
            "total_tokens_generated": result.total_tokens_generated,
        },
        "prompts": [
            {
                "prompt": pr.prompt,
                "output": pr.output,
                "drafted": pr.stats.drafted_tokens,
                "accepted": pr.stats.accepted_tokens,
                "rejected": pr.stats.rejected_tokens,
                "draft_calls": pr.stats.draft_calls,
                "target_calls": pr.stats.target_calls,
                "time": pr.stats.generation_time,
            }
            for pr in result.prompt_results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def progress_printer(current: int, total: int) -> None:
    """Simple progress callback that prints to stdout."""
    print(f"\r  [{current}/{total}]", end="", flush=True)
    if current == total:
        print()
