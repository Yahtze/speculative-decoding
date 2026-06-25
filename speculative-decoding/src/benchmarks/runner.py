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
        save_name="experiment_001",  # saves to results/experiment_001.json + .csv
    )
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Callable, Optional

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    tqdm = None

from ..models.wrapper import LanguageModel
from ..decoding.speculative import SpeculativeDecoder, DecodeStats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


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
    total_tokenization_time: float = 0.0
    total_generation_time: float = 0.0
    total_decode_time: float = 0.0
    total_tokens_generated: int = 0

    # Per-prompt results
    prompt_results: list[PromptResult] = field(default_factory=list)

    # Configuration
    draft_model_name: str = ""
    target_model_name: str = ""
    k: int = 0
    max_new_tokens: int = 0

    # Runtime timings outside prompt loop
    runtime_timings: dict[str, float] = field(default_factory=dict)

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

    @property
    def total_prompt_time(self) -> float:
        """End-to-end time across all prompts."""
        return (
            self.total_tokenization_time
            + self.total_generation_time
            + self.total_decode_time
        )

    @property
    def avg_tokenization_time(self) -> float:
        """Average tokenization time per prompt."""
        if self.total_prompts == 0:
            return 0.0
        return self.total_tokenization_time / self.total_prompts

    @property
    def avg_generation_time(self) -> float:
        """Average inference/generation time per prompt."""
        if self.total_prompts == 0:
            return 0.0
        return self.total_generation_time / self.total_prompts

    @property
    def avg_decode_time(self) -> float:
        """Average decode-to-text time per prompt."""
        if self.total_prompts == 0:
            return 0.0
        return self.total_decode_time / self.total_prompts

    @property
    def avg_prompt_time(self) -> float:
        """Average end-to-end time per prompt."""
        if self.total_prompts == 0:
            return 0.0
        return self.total_prompt_time / self.total_prompts

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
            f"Avg tokenization:{self.avg_tokenization_time:.3f} s/prompt\n"
            f"Avg inference:   {self.avg_generation_time:.3f} s/prompt\n"
            f"Avg decode:      {self.avg_decode_time:.3f} s/prompt\n"
            f"Avg total/prompt:{self.avg_prompt_time:.3f} s/prompt\n"
            f"Target calls:    {self.target_calls:.1f} avg/prompt\n"
            f"Tokens/call:     {self.tokens_per_target_call:.2f}\n"
            f"{'-' * 40}\n"
            f"Total drafted:   {self.total_drafted_tokens}\n"
            f"Total accepted:  {self.total_accepted_tokens}\n"
            f"Total rejected:  {self.total_rejected_tokens}\n"
            f"Total draft calls:  {self.total_draft_calls}\n"
            f"Total target calls: {self.total_target_calls}\n"
            f"Total tokenization: {self.total_tokenization_time:.2f}s\n"
            f"Total inference: {self.total_generation_time:.2f}s\n"
            f"Total decode:    {self.total_decode_time:.2f}s\n"
            f"Total time:      {self.total_prompt_time:.2f}s\n"
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
    timing_progress: bool = False,
    save_name: Optional[str] = None,
    results_dir: str | Path = RESULTS_DIR,
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
        timing_progress: If True, show tqdm progress with per-step timing breakdown.
        save_name: If set, save results to {results_dir}/{save_name}.json and .csv.
        results_dir: Directory to save results. Defaults to speculative-decoding/results/.

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

    progress_bar = None
    if timing_progress and tqdm is not None:
        progress_bar = tqdm(total=len(prompts), desc="SpecDec", unit="prompt", file=sys.stdout)

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
        result.total_tokenization_time += stats.tokenization_time
        result.total_generation_time += stats.generation_time
        result.total_decode_time += stats.decode_time
        result.total_tokens_generated += (stats.accepted_tokens + stats.rejected_tokens)

        if progress_bar is not None:
            progress_bar.set_postfix({
                "tok": f"{stats.tokenization_time:.3f}s",
                "infer": f"{stats.generation_time:.3f}s",
                "decode": f"{stats.decode_time:.3f}s",
                "total": f"{stats.total_time:.3f}s",
            })
            progress_bar.update(1)

        # Progress callback
        if progress_fn:
            progress_fn(i + 1, len(prompts))

    if progress_bar is not None:
        progress_bar.close()

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

    # Auto-save if save_name provided
    if save_name:
        save_dir = Path(results_dir)
        save_result_json(result, save_dir / f"{save_name}.json")
        save_result_csv(result, save_dir / f"{save_name}.csv")

    return result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_result_json(result: ExperimentResult, path: str | Path) -> None:
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
            "avg_tokenization_time": result.avg_tokenization_time,
            "avg_generation_time": result.avg_generation_time,
            "avg_decode_time": result.avg_decode_time,
            "avg_prompt_time": result.avg_prompt_time,
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
            "total_tokenization_time": result.total_tokenization_time,
            "total_generation_time": result.total_generation_time,
            "total_decode_time": result.total_decode_time,
            "total_prompt_time": result.total_prompt_time,
            "total_tokens_generated": result.total_tokens_generated,
        },
        "runtime_timings": result.runtime_timings,
        "prompts": [
            {
                "prompt": pr.prompt,
                "output": pr.output,
                "drafted": pr.stats.drafted_tokens,
                "accepted": pr.stats.accepted_tokens,
                "rejected": pr.stats.rejected_tokens,
                "draft_calls": pr.stats.draft_calls,
                "target_calls": pr.stats.target_calls,
                "tokenization_time": pr.stats.tokenization_time,
                "generation_time": pr.stats.generation_time,
                "decode_time": pr.stats.decode_time,
                "total_time": pr.stats.total_time,
            }
            for pr in result.prompt_results
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_result_csv(result: ExperimentResult, path: str | Path) -> None:
    """Save per-prompt results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt",
        "output",
        "drafted_tokens",
        "accepted_tokens",
        "rejected_tokens",
        "acceptance_rate",
        "draft_calls",
        "target_calls",
        "tokenization_time",
        "generation_time",
        "decode_time",
        "total_time",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pr in result.prompt_results:
            writer.writerow({
                "prompt": pr.prompt,
                "output": pr.output,
                "drafted_tokens": pr.stats.drafted_tokens,
                "accepted_tokens": pr.stats.accepted_tokens,
                "rejected_tokens": pr.stats.rejected_tokens,
                "acceptance_rate": f"{pr.stats.acceptance_rate:.4f}",
                "draft_calls": pr.stats.draft_calls,
                "target_calls": pr.stats.target_calls,
                "tokenization_time": f"{pr.stats.tokenization_time:.4f}",
                "generation_time": f"{pr.stats.generation_time:.4f}",
                "decode_time": f"{pr.stats.decode_time:.4f}",
                "total_time": f"{pr.stats.total_time:.4f}",
            })


# Backward-compatible alias
def save_result(result: ExperimentResult, path: str | Path) -> None:
    """Save experiment result to JSON (backward-compatible alias)."""
    save_result_json(result, path)


def progress_printer(current: int, total: int) -> None:
    """Simple progress callback that prints to stdout."""
    print(f"\r  [{current}/{total}]", end="", flush=True)
    if current == total:
        print()
