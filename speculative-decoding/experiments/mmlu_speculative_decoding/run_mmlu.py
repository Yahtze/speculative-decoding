#!/usr/bin/env python3
"""
MMLU Speculative Decoding Experiment Runner

Evaluates speculative decoding on MMLU multiple-choice questions.
Loads config from config.yml in this directory.

Usage:
    cd speculative-decoding
    python experiments/mmlu/run_mmlu.py
    
    # Or with custom config:
    python experiments/mmlu/run_mmlu.py --config experiments/mmlu/config.yml
"""

import sys
import argparse
from pathlib import Path

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # experiments/mmlu -> experiments -> speculative-decoding
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from datasets import load_dataset

from src.models.wrapper import LanguageModel
from src.benchmarks.runner import (
    run_experiment,
    progress_printer,
    save_result_json,
    save_result_csv,
)


# ---------------------------------------------------------------------------
# MMLU prompt formatting
# ---------------------------------------------------------------------------

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """
    Format an MMLU question as a multiple-choice prompt.
    
    Example format:
        Question: What is the capital of France?
        A. London
        B. Paris
        C. Berlin
        D. Madrid
        Answer:
    """
    choices_str = "\n".join(
        f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, choices)
    )
    return f"Question: {question}\n{choices_str}\nAnswer:"


def load_mmlu_prompts(task: str, dataset_name: str = "cais/mmlu", max_questions: int | None = None) -> tuple[list[str], list[int]]:
    """
    Load MMLU questions and format as prompts.
    
    Returns:
        (prompts, answer_indices) - formatted prompts and ground truth answer indices
    """
    print(f"Loading MMLU dataset: {dataset_name}, task: {task}")
    ds = load_dataset(dataset_name, task, split="test")
    
    if max_questions:
        ds = ds.select(range(min(max_questions, len(ds))))
    
    prompts = []
    answer_indices = []
    
    for example in ds:
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]  # 0-indexed into choices
        
        prompt = format_mmlu_prompt(question, choices)
        prompts.append(prompt)
        answer_indices.append(answer_idx)
    
    print(f"Loaded {len(prompts)} questions")
    return prompts, answer_indices


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(generated_text: str, prompt: str) -> str | None:
    """
    Extract the answer letter from generated text.
    
    Looks for the first occurrence of A, B, C, or D after the prompt.
    """
    # Get only the generated portion
    response = generated_text[len(prompt):].strip()
    
    if not response:
        return None
    
    # Look for first answer letter
    for char in response[:10]:  # Check first 10 chars
        if char.upper() in CHOICE_LABELS:
            return char.upper()
    
    return None


def evaluate_accuracy(prompt_results: list, answer_indices: list[int], prompts: list[str]) -> dict:
    """
    Compute accuracy metrics from experiment results.
    
    Returns dict with accuracy stats.
    """
    correct = 0
    invalid = 0
    total = len(prompt_results)
    
    for i, (result, true_idx) in enumerate(zip(prompt_results, answer_indices)):
        predicted = extract_answer(result.output, prompts[i])
        
        if predicted is None:
            invalid += 1
        else:
            predicted_idx = CHOICE_LABELS.index(predicted)
            if predicted_idx == true_idx:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    valid_rate = (total - invalid) / total if total > 0 else 0.0
    
    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": accuracy,
        "valid_format_rate": valid_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MMLU Speculative Decoding Experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override MMLU task from config",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Override max questions from config",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Override draft model name",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Override target model name",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override k (tokens per draft round)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose speculative decoding output",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply CLI overrides
    task = args.task or config["mmlu"]["task"]
    max_questions = args.max_questions or config["mmlu"].get("max_questions")
    draft_model_name = args.draft_model or config["models"]["draft"]["name"]
    target_model_name = args.target_model or config["models"]["target"]["name"]
    k = args.k or config["decoding"]["k"]
    max_new_tokens = config["decoding"]["max_new_tokens"]
    temperature = config["decoding"]["temperature"]
    top_k = config["decoding"]["top_k"]
    verbose = args.verbose or config["output"].get("verbose", False)
    
    # Results path
    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    save_name = f"mmlu_{task}"
    
    # Print experiment config
    print("=" * 60)
    print(f"MMLU Speculative Decoding Experiment")
    print("=" * 60)
    print(f"Task:           {task}")
    print(f"Draft model:    {draft_model_name}")
    print(f"Target model:   {target_model_name}")
    print(f"k:              {k}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature:    {temperature}")
    print(f"Top-k:          {top_k}")
    print(f"Results dir:    {results_dir}")
    print(f"Save name:      {save_name}")
    print("=" * 60)
    
    # Load MMLU prompts
    prompts, answer_indices = load_mmlu_prompts(
        task=task,
        dataset_name=config["mmlu"]["dataset"],
        max_questions=max_questions,
    )
    
    # Load models
    print(f"\nLoading draft model: {draft_model_name}")
    draft_model = LanguageModel(draft_model_name)
    
    print(f"Loading target model: {target_model_name}")
    target_model = LanguageModel(target_model_name)
    
    # Run experiment
    print(f"\nRunning speculative decoding on {len(prompts)} questions...")
    
    result = run_experiment(
        draft_model=draft_model,
        target_model=target_model,
        k=k,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        verbose=verbose,
        progress_fn=progress_printer if config["output"].get("print_progress", True) else None,
        save_name=save_name,
        results_dir=results_dir,
    )
    
    # Evaluate accuracy
    accuracy_metrics = evaluate_accuracy(result.prompt_results, answer_indices, prompts)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(result.summary())
    print("-" * 60)
    print(f"Accuracy:         {accuracy_metrics['accuracy']:.1%}")
    print(f"Valid format:     {accuracy_metrics['valid_format_rate']:.1%}")
    print(f"Correct:          {accuracy_metrics['correct']}/{accuracy_metrics['total']}")
    print(f"Invalid format:   {accuracy_metrics['invalid']}")
    print("=" * 60)
    
    # Save accuracy metrics alongside results
    import json
    accuracy_path = results_dir / f"{save_name}_accuracy.json"
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(accuracy_path, "w") as f:
        json.dump({
            "task": task,
            "model_config": {
                "draft": draft_model_name,
                "target": target_model_name,
                "k": k,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
            },
            "accuracy": accuracy_metrics,
            "decoding_stats": {
                "acceptance_rate": result.acceptance_rate,
                "throughput": result.throughput,
                "latency": result.latency,
            },
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  {results_dir / save_name}.json")
    print(f"  {results_dir / save_name}.csv")
    print(f"  {accuracy_path}")


if __name__ == "__main__":
    main()
