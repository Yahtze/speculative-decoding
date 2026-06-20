#!/usr/bin/env python3
"""
MMLU Baseline Experiment — Standard Greedy Decoding

Runs the target model with greedy decoding (no speculative decoding).
Used as baseline to compare against speculative decoding experiments.

Usage:
    cd speculative-decoding
    python experiments/mmlu_baseline/run_mmlu_baseline.py
    python experiments/mmlu_baseline/run_mmlu_baseline.py --task astronomy --max-questions 50
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # experiments/mmlu_baseline -> experiments -> speculative-decoding
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from datasets import load_dataset

from src.models.wrapper import LanguageModel
from src.decoding.greedy import greedy_decode


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CHOICE_LABELS = ["A", "B", "C", "D"]


@dataclass
class PromptResult:
    """Result for a single prompt."""
    prompt: str
    output: str
    generation_time: float
    tokens_generated: int


@dataclass
class BaselineResult:
    """Aggregated baseline results."""
    model_name: str = ""
    max_new_tokens: int = 0
    total_prompts: int = 0
    total_generation_time: float = 0.0
    total_tokens_generated: int = 0
    prompt_results: list[PromptResult] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Tokens per second."""
        if self.total_generation_time == 0:
            return 0.0
        return self.total_tokens_generated / self.total_generation_time

    @property
    def latency(self) -> float:
        """Seconds per prompt."""
        if self.total_prompts == 0:
            return 0.0
        return self.total_generation_time / self.total_prompts

    def summary(self) -> str:
        return (
            f"Baseline Result (Greedy Decoding)\n"
            f"{'=' * 40}\n"
            f"Model:           {self.model_name}\n"
            f"Max new tokens:  {self.max_new_tokens}\n"
            f"Prompts:         {self.total_prompts}\n"
            f"{'-' * 40}\n"
            f"Throughput:      {self.throughput:.2f} tokens/s\n"
            f"Latency:         {self.latency:.3f} s/prompt\n"
            f"{'-' * 40}\n"
            f"Total time:      {self.total_generation_time:.2f}s\n"
            f"Total tokens:    {self.total_tokens_generated}\n"
        )


# ---------------------------------------------------------------------------
# MMLU loading (same as speculative experiment)
# ---------------------------------------------------------------------------

def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    choices_str = "\n".join(
        f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, choices)
    )
    return f"Question: {question}\n{choices_str}\nAnswer:"


def load_mmlu_prompts(task: str, dataset_name: str = "cais/mmlu", max_questions: int | None = None):
    print(f"Loading MMLU dataset: {dataset_name}, task: {task}")
    ds = load_dataset(dataset_name, task, split="test")

    if max_questions:
        ds = ds.select(range(min(max_questions, len(ds))))

    prompts = []
    answer_indices = []

    for example in ds:
        prompt = format_mmlu_prompt(example["question"], example["choices"])
        prompts.append(prompt)
        answer_indices.append(example["answer"])

    print(f"Loaded {len(prompts)} questions")
    return prompts, answer_indices


# ---------------------------------------------------------------------------
# Answer extraction & accuracy
# ---------------------------------------------------------------------------

def extract_answer(generated_text: str, prompt: str) -> str | None:
    response = generated_text[len(prompt):].strip()
    if not response:
        return None
    for char in response[:10]:
        if char.upper() in CHOICE_LABELS:
            return char.upper()
    return None


def evaluate_accuracy(prompt_results: list[PromptResult], answer_indices: list[int]) -> dict:
    correct = 0
    invalid = 0
    total = len(prompt_results)

    for i, (result, true_idx) in enumerate(zip(prompt_results, answer_indices)):
        predicted = extract_answer(result.output, result.prompt)
        if predicted is None:
            invalid += 1
        else:
            if CHOICE_LABELS.index(predicted) == true_idx:
                correct += 1

    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total > 0 else 0.0,
        "valid_format_rate": (total - invalid) / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baseline(
    model: LanguageModel,
    prompts: list[str],
    max_new_tokens: int,
    progress_fn=None,
) -> BaselineResult:
    result = BaselineResult(
        model_name=repr(model),
        max_new_tokens=max_new_tokens,
        total_prompts=len(prompts),
    )

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        token_ids, output = greedy_decode(model, prompt, max_new_tokens)
        elapsed = time.perf_counter() - start

        tokens_generated = len(token_ids) - len(model.encode(prompt))

        result.prompt_results.append(PromptResult(
            prompt=prompt,
            output=output,
            generation_time=elapsed,
            tokens_generated=tokens_generated,
        ))

        result.total_generation_time += elapsed
        result.total_tokens_generated += tokens_generated

        if progress_fn:
            progress_fn(i + 1, len(prompts))

    return result


def progress_printer(current: int, total: int):
    print(f"\r  [{current}/{total}]", end="", flush=True)
    if current == total:
        print()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(result: BaselineResult, accuracy: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config": {
            "model": result.model_name,
            "max_new_tokens": result.max_new_tokens,
            "decoding": "greedy",
        },
        "metrics": {
            "throughput": result.throughput,
            "latency": result.latency,
        },
        "accuracy": accuracy,
        "totals": {
            "total_prompts": result.total_prompts,
            "total_generation_time": result.total_generation_time,
            "total_tokens_generated": result.total_tokens_generated,
        },
        "prompts": [
            {
                "prompt": pr.prompt,
                "output": pr.output,
                "generation_time": pr.generation_time,
                "tokens_generated": pr.tokens_generated,
            }
            for pr in result.prompt_results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MMLU Baseline (Greedy Decoding)")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--target-model", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = args.task or config["mmlu"]["task"]
    max_questions = args.max_questions or config["mmlu"].get("max_questions")
    target_model_name = args.target_model or config["models"]["target"]["name"]
    max_new_tokens = config["decoding"]["max_new_tokens"]

    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    save_name = f"mmlu_baseline_{task}"

    print("=" * 60)
    print("MMLU Baseline Experiment (Greedy Decoding)")
    print("=" * 60)
    print(f"Task:           {task}")
    print(f"Model:          {target_model_name}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Results:        {results_dir / save_name}.json")
    print("=" * 60)

    prompts, answer_indices = load_mmlu_prompts(
        task=task,
        dataset_name=config["mmlu"]["dataset"],
        max_questions=max_questions,
    )

    print(f"\nLoading model: {target_model_name}")
    model = LanguageModel(target_model_name)

    print(f"\nRunning greedy decoding on {len(prompts)} questions...")
    result = run_baseline(
        model=model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        progress_fn=progress_printer if config["output"].get("print_progress", True) else None,
    )

    accuracy = evaluate_accuracy(result.prompt_results, answer_indices)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(result.summary())
    print("-" * 60)
    print(f"Accuracy:         {accuracy['accuracy']:.1%}")
    print(f"Valid format:     {accuracy['valid_format_rate']:.1%}")
    print(f"Correct:          {accuracy['correct']}/{accuracy['total']}")
    print(f"Invalid format:   {accuracy['invalid']}")
    print("=" * 60)

    output_path = results_dir / f"{save_name}.json"
    save_results(result, accuracy, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
