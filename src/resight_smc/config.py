from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_bool(value: str | bool) -> bool:
    """Parse booleans in the same ``--parameter value`` form as other options."""

    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Store backbone, SMC, visual-scout, and answer-readout settings."""

    # Task and output
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dataset: str = "logicvista"
    split: str | None = None
    method: str = "resight"
    prompt_mode: str = "cot"
    seed: int = 0
    max_samples: int | None = None
    start_index: int = 0
    output_dir: str = "outputs"
    run_id: str | None = None

    # Model runtime
    engine: str = "transformers"
    dtype: str = "bfloat16"
    device: str = "cuda"
    device_map: str | None = None
    attention_backend: str = "sdpa"
    min_pixels: int = 200704
    max_pixels: int = 1003520
    compile: bool = False
    baseline_batch_size: int = 8
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 4096

    # Ordinary sampling
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # Power-SMC
    max_new_tokens: int = 1024
    alpha: float = 2.0
    # None uses gamma=2 for ReSight and gamma=1 for Power/Island-SMC.
    gamma: float | None = None
    alpha_ramp_tokens: int = 128
    num_particles: int = 32
    num_islands: int = 4
    particles_per_island: int | None = None
    ess_threshold: float = 0.5
    checkpoint_interval: int = 32

    # Visual scouts
    scout_fraction: float = 0.25
    visual_checkpoint: int = 40
    visual_length: int = 16
    # Bias all original-image tokens, then add a region-specific bias.
    image_attention_logit_bias: float = 0.6931471805599453
    region_attention_logit_bias: float = 1.3862943611198906
    # (0, 1) specifies relative depth; integers specify layer indices, with -1 last.
    attention_layer: float = -1
    area_exponent: float = 0.75
    overlap_penalty: float = 1.0
    overlap_penalty_scope: str = "global"

    # Prompt
    cot_suffix: str = "Think step by step and end with `Final answer: ...`."
    direct_suffix: str = "Answer directly and end with `Final answer: ...`."

    def finalize(self) -> None:
        """Resolve method defaults and validate population, scout, and attention settings."""

        if self.run_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            model_name = self.model.replace("/", "_")
            self.run_id = f"{model_name}_{self.dataset}_{self.method}_s{self.seed}_{stamp}"
        if self.method == "base":
            self.prompt_mode = "direct"
        elif self.method == "cot":
            self.prompt_mode = "cot"
        if self.method == "power_smc":
            self.num_islands = 1
            if self.particles_per_island is not None:
                self.num_particles = self.particles_per_island
        elif self.particles_per_island is not None:
            self.num_particles = self.num_islands * self.particles_per_island

        if self.gamma is None:
            self.gamma = 2.0 if self.method == "resight" else 1.0

        # These constraints are required by the SMC and scout definitions.
        if self.num_particles % self.num_islands != 0:
            raise ValueError("num-particles must be divisible by num-islands.")
        self.particles_per_island = self.num_particles // self.num_islands
        if self.method == "resight":
            if self.visual_length < 1:
                raise ValueError("attention-reactivated scouts require visual-length >= 1.")
            if self.attention_backend == "flash_attention_2":
                raise ValueError(
                    "attention-reactivated scouts require an additive-bias-compatible "
                    "sdpa or eager backend."
                )
            if self.visual_checkpoint < 1:
                raise ValueError("attention-reactivated scouts require visual-checkpoint >= 1.")
            visual_end = self.visual_checkpoint + self.visual_length
            if visual_end > self.max_new_tokens:
                raise ValueError("the visual episode must end within max-new-tokens.")
            next_checkpoint = (
                self.visual_checkpoint // self.checkpoint_interval + 1
            ) * self.checkpoint_interval
            if visual_end > next_checkpoint:
                raise ValueError("the visual episode cannot cross the next SMC checkpoint.")
        if self.image_attention_logit_bias < 0.0:
            raise ValueError("image-attention-logit-bias must be nonnegative.")
        if self.region_attention_logit_bias < 0.0:
            raise ValueError("region-attention-logit-bias must be nonnegative.")
        if self.overlap_penalty_scope not in {"global", "island"}:
            raise ValueError("overlap-penalty-scope must be global or island.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive.")

    @property
    def run_dir(self) -> Path:
        """Return the experiment directory determined by ``output_dir`` and ``run_id``."""

        assert self.run_id is not None
        return Path(self.output_dir) / self.run_id

    def to_dict(self) -> dict[str, Any]:
        """Export the complete serializable experiment configuration."""

        return asdict(self)


def _option_name(field_name: str) -> str:
    """Convert a dataclass field name to its hyphenated CLI option."""

    return "--" + field_name.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options for all configurable fields."""

    parser = argparse.ArgumentParser(description="Run ReSight-SMC experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(_option_name("model"))
    parser.add_argument(_option_name("dataset"))
    parser.add_argument(_option_name("split"))
    parser.add_argument(
        _option_name("method"), choices=["base", "cot", "power_smc", "island_smc", "resight"]
    )
    parser.add_argument(_option_name("prompt_mode"), choices=["direct", "cot"])
    parser.add_argument(_option_name("seed"), type=int)
    parser.add_argument(_option_name("max_samples"), type=int)
    parser.add_argument(_option_name("start_index"), type=int)
    parser.add_argument(_option_name("output_dir"))
    parser.add_argument(_option_name("run_id"))

    parser.add_argument(_option_name("engine"), choices=["transformers", "vllm"])
    parser.add_argument(_option_name("dtype"), choices=["bfloat16", "float16", "float32"])
    parser.add_argument(_option_name("device"))
    parser.add_argument(_option_name("device_map"))
    parser.add_argument(
        _option_name("attention_backend"), choices=["sdpa", "eager", "flash_attention_2"]
    )
    parser.add_argument(_option_name("min_pixels"), type=int)
    parser.add_argument(_option_name("max_pixels"), type=int)
    parser.add_argument(_option_name("compile"), type=parse_bool)
    parser.add_argument(_option_name("baseline_batch_size"), type=int)
    parser.add_argument(_option_name("gpu_memory_utilization"), type=float)
    parser.add_argument(_option_name("tensor_parallel_size"), type=int)
    parser.add_argument(_option_name("max_model_len"), type=int)

    parser.add_argument(_option_name("temperature"), type=float)
    parser.add_argument(_option_name("top_p"), type=float)
    parser.add_argument(_option_name("top_k"), type=int)
    parser.add_argument(_option_name("max_new_tokens"), type=int)
    parser.add_argument(_option_name("alpha"), type=float)
    parser.add_argument(_option_name("gamma"), type=float)
    parser.add_argument(_option_name("alpha_ramp_tokens"), type=int)
    parser.add_argument(_option_name("num_particles"), type=int)
    parser.add_argument(_option_name("num_islands"), type=int)
    parser.add_argument(_option_name("particles_per_island"), type=int)
    parser.add_argument(_option_name("ess_threshold"), type=float)
    parser.add_argument(_option_name("checkpoint_interval"), type=int)

    parser.add_argument(_option_name("scout_fraction"), type=float)
    parser.add_argument(_option_name("visual_checkpoint"), type=int)
    parser.add_argument(_option_name("visual_length"), type=int)
    parser.add_argument(_option_name("image_attention_logit_bias"), type=float)
    parser.add_argument(_option_name("region_attention_logit_bias"), type=float)
    parser.add_argument(_option_name("attention_layer"), type=float)
    parser.add_argument(_option_name("area_exponent"), type=float)
    parser.add_argument(_option_name("overlap_penalty"), type=float)
    parser.add_argument(
        _option_name("overlap_penalty_scope"), choices=["global", "island"]
    )
    parser.add_argument(_option_name("cot_suffix"))
    parser.add_argument(_option_name("direct_suffix"))
    return parser


def load_config(argv: list[str] | None = None) -> Config:
    """Merge YAML and explicit CLI overrides, then finalize the configuration."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    valid = {item.name for item in fields(Config)}
    values = {key: value for key, value in values.items() if key in valid}
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            values[key] = value
    cfg = Config(**values)
    cfg.finalize()
    return cfg
