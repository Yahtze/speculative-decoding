from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "src/benchmarks/runner.py"


@dataclass
class FakeDecodeStats:
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    draft_calls: int = 0
    target_calls: int = 0
    generation_time: float = 0.0
    tokenization_time: float = 0.0
    decode_time: float = 0.0

    @property
    def total_time(self) -> float:
        return self.tokenization_time + self.generation_time + self.decode_time

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens

    @property
    def rejection_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.rejected_tokens / self.drafted_tokens


class FakeSpeculativeDecoder:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def generate(self, prompt: str, max_new_tokens: int):
        self.calls += 1
        stats = FakeDecodeStats(
            drafted_tokens=4,
            accepted_tokens=3,
            rejected_tokens=1,
            draft_calls=2,
            target_calls=2,
            generation_time=0.2,
            tokenization_time=0.05,
            decode_time=0.01,
        )
        return [1, 2, 3], f"out:{prompt}:{max_new_tokens}", stats


class FakeLanguageModel:
    def __repr__(self) -> str:
        return "FakeLanguageModel()"


def load_runner_module():
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(ROOT / "src")]
    sys.modules["src"] = src_pkg

    benchmarks_pkg = types.ModuleType("src.benchmarks")
    benchmarks_pkg.__path__ = [str(ROOT / "src/benchmarks")]
    sys.modules["src.benchmarks"] = benchmarks_pkg

    models_pkg = types.ModuleType("src.models")
    models_pkg.__path__ = [str(ROOT / "src/models")]
    sys.modules["src.models"] = models_pkg

    wrapper_mod = types.ModuleType("src.models.wrapper")
    wrapper_mod.LanguageModel = FakeLanguageModel
    sys.modules["src.models.wrapper"] = wrapper_mod

    decoding_pkg = types.ModuleType("src.decoding")
    decoding_pkg.__path__ = [str(ROOT / "src/decoding")]
    sys.modules["src.decoding"] = decoding_pkg

    speculative_mod = types.ModuleType("src.decoding.speculative")
    speculative_mod.SpeculativeDecoder = FakeSpeculativeDecoder
    speculative_mod.DecodeStats = FakeDecodeStats
    sys.modules["src.decoding.speculative"] = speculative_mod

    spec = importlib.util.spec_from_file_location("src.benchmarks.runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.benchmarks.runner"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load_runner_module()


def test_run_experiment_aggregates_timing_fields():
    result = runner.run_experiment(
        draft_model=FakeLanguageModel(),
        target_model=FakeLanguageModel(),
        k=3,
        prompts=["a", "b"],
        max_new_tokens=5,
    )

    assert result.total_prompts == 2
    assert result.total_tokenization_time == 0.1
    assert result.total_decode_time == 0.02
    assert result.total_prompt_time == 0.52
    assert result.avg_tokenization_time == 0.05
    assert result.avg_generation_time == 0.2
    assert result.avg_decode_time == 0.01
    assert result.avg_prompt_time == 0.26

    summary = result.summary()
    assert "Avg tokenization:" in summary
    assert "Avg inference:" in summary
    assert "Avg total/prompt:" in summary


def test_save_result_json_includes_timing_breakdown(tmp_path: Path):
    result = runner.run_experiment(
        draft_model=FakeLanguageModel(),
        target_model=FakeLanguageModel(),
        k=3,
        prompts=["only"],
        max_new_tokens=7,
    )
    result.runtime_timings = {
        "dataset_load_time": 1.25,
        "draft_model_load_time": 2.5,
        "target_model_load_time": 3.75,
        "experiment_wall_time": 4.0,
        "accuracy_eval_time": 0.5,
    }

    out = tmp_path / "result.json"
    runner.save_result_json(result, out)
    data = json.loads(out.read_text())

    assert data["metrics"]["avg_tokenization_time"] == 0.05
    assert data["metrics"]["avg_generation_time"] == 0.2
    assert data["metrics"]["avg_decode_time"] == 0.01
    assert data["metrics"]["avg_prompt_time"] == 0.26
    assert data["totals"]["total_tokenization_time"] == 0.05
    assert data["totals"]["total_decode_time"] == 0.01
    assert data["runtime_timings"]["dataset_load_time"] == 1.25
    assert data["runtime_timings"]["draft_model_load_time"] == 2.5
    assert data["runtime_timings"]["target_model_load_time"] == 3.75
    assert data["runtime_timings"]["experiment_wall_time"] == 4.0
    assert data["runtime_timings"]["accuracy_eval_time"] == 0.5
    assert data["prompts"][0]["tokenization_time"] == 0.05
    assert data["prompts"][0]["generation_time"] == 0.2
    assert data["prompts"][0]["decode_time"] == 0.01
    assert data["prompts"][0]["total_time"] == 0.26


if __name__ == "__main__":
    test_run_experiment_aggregates_timing_fields()
    test_save_result_json_includes_timing_breakdown(Path("/tmp"))
    print("All tests passed!")
