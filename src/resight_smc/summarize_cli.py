from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_jsonl


def _load_runs(paths: list[str]) -> list[dict[str, dict[str, Any]]]:
    return [
        {record["sample_id"]: record for record in read_jsonl(Path(path) / "predictions.jsonl")}
        for path in paths
    ]


def _score(record: dict[str, Any]) -> float:
    return float(record["evaluation"]["score"])


def _summary(runs: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    sample_ids = sorted(set.intersection(*(set(run) for run in runs)))
    per_example = np.array([[_score(run[uid]) for run in runs] for uid in sample_ids])
    diagnostics = [run[uid]["diagnostics"] for run in runs for uid in sample_ids]
    return {
        "num_examples": len(sample_ids),
        "num_seeds": len(runs),
        "mean_accuracy": float(per_example.mean()),
        "accuracy_by_seed": [float(per_example[:, index].mean()) for index in range(len(runs))],
        "pass_at_4": float(np.any(per_example[:, :4] > 0, axis=1).mean())
        if len(runs) >= 4
        else None,
        "mean_latency_seconds": float(np.mean([item["latency_seconds"] for item in diagnostics])),
        "mean_coverage_at_32": (
            float(np.mean([item["coverage_at_32"] for item in diagnostics]))
            if all("coverage_at_32" in item for item in diagnostics)
            else None
        ),
        "mean_effective_roots": (
            float(np.mean([item["final_effective_roots"] for item in diagnostics]))
            if all("final_effective_roots" in item for item in diagnostics)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize independent seed runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = _summary(_load_runs(args.runs))
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
