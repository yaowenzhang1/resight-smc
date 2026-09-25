from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import Config


class ArtifactWriter:
    """Store each run in one directory, preserving per-question results if interrupted."""

    def __init__(self, cfg: Config):
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.prediction_path = self.run_dir / "predictions.jsonl"

    @staticmethod
    def _append(path: Path, value: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()

    def prediction(self, value: dict[str, Any]) -> None:
        self._append(self.prediction_path, value)

    def metrics(self, value: dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
