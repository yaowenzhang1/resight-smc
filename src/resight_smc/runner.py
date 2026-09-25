from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .artifacts import ArtifactWriter
from .baselines import BaselineResult, TransformersBaseline, VLLMBaseline
from .config import Config
from .data import Example, format_question, load_examples
from .evaluate import aggregate, canonical_answer, evaluate
from .model import TransformersVLM
from .smc import PowerSMC, SMCResult


def _metrics(records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Aggregate task metrics, latency, VRAM, genealogy, and visual-cost diagnostics."""

    metrics = aggregate(records)
    diagnostics = [record["diagnostics"] for record in records]
    latencies = [float(item["latency_seconds"]) for item in diagnostics]
    metrics.update(
        {
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "mean_peak_vram_gb": sum(float(item["peak_vram_gb"]) for item in diagnostics)
            / len(diagnostics),
            "coverage_at_32": (
                sum(bool(item.get("coverage_at_32")) for item in diagnostics) / len(diagnostics)
                if any("coverage_at_32" in item for item in diagnostics)
                else None
            ),
            "mean_effective_roots": (
                sum(float(item["final_effective_roots"]) for item in diagnostics) / len(diagnostics)
                if all("final_effective_roots" in item for item in diagnostics)
                else None
            ),
            "mean_encoded_visual_tokens": (
                sum(float(item["total_encoded_visual_tokens"]) for item in diagnostics)
                / len(diagnostics)
                if all("total_encoded_visual_tokens" in item for item in diagnostics)
                else None
            ),
            "mean_answer_power_seconds": (
                sum(float(item["answer_power_seconds"]) for item in diagnostics)
                / len(diagnostics)
                if all("answer_power_seconds" in item for item in diagnostics)
                else None
            ),
            "run_dir": str(run_dir.resolve()),
        }
    )
    return metrics


def _baseline_record(
    cfg: Config, example: Example, result: BaselineResult
) -> dict[str, Any]:
    """Evaluate a baseline response and store it in the shared artifact schema."""

    evaluation = evaluate(cfg.dataset, example, result.text)
    record = {
        "sample_id": example.uid,
        "category": example.category,
        "question": example.question,
        "references": example.answers,
        "answer": result.text,
        "answer_tokens": result.tokens,
        "evaluation": evaluation,
        "diagnostics": {
            "latency_seconds": result.latency_seconds,
            "peak_vram_gb": result.peak_vram_gb,
        },
    }
    return record


def _smc_record(
    cfg: Config,
    example: Example,
    result: SMCResult,
    answer_rng: random.Random,
) -> dict[str, Any]:
    """Perform the second-stage answer-marginal power readout.

    Aggregate first-stage particle mass by canonical answer to obtain
    ``mu_hat_alpha(a)``, then sample from
    ``q_hat(a) proportional to mu_hat_alpha(a)**gamma``. For ``gamma != 1``,
    draw a supporting trajectory by its original particle mass within that answer.
    """

    readout_start = time.perf_counter()
    canonical_answers = [
        canonical_answer(cfg.dataset, example, text) for text in result.particle_texts
    ]
    answer_masses: dict[str, float] = {}
    answer_particles: dict[str, list[int]] = {}
    for index, (answer, weight) in enumerate(
        zip(canonical_answers, result.terminal_weights, strict=True)
    ):
        answer_masses[answer] = answer_masses.get(answer, 0.0) + float(weight)
        answer_particles.setdefault(answer, []).append(index)

    answers = [answer for answer, mass in answer_masses.items() if mass > 0.0]
    log_answer_mass = [cfg.gamma * math.log(answer_masses[answer]) for answer in answers]
    max_log_mass = max(log_answer_mass)
    powered_mass = [math.exp(value - max_log_mass) for value in log_answer_mass]
    powered_total = sum(powered_mass)
    answer_probabilities = [value / powered_total for value in powered_mass]

    old_particle = (
        int(result.diagnostics["chosen_island"])
        * int(result.diagnostics["particles_per_island"])
        + int(result.diagnostics["chosen_particle"])
    )
    selected_particle = old_particle
    if cfg.gamma != 1.0:
        threshold = answer_rng.random()
        cumulative = 0.0
        selected_answer = answers[-1]
        for answer, probability in zip(answers, answer_probabilities, strict=True):
            cumulative += probability
            if threshold <= cumulative:
                selected_answer = answer
                break

        particle_indices = answer_particles[selected_answer]
        selected_mass = answer_masses[selected_answer]
        threshold = answer_rng.random()
        cumulative = 0.0
        selected_particle = particle_indices[-1]
        for index in particle_indices:
            cumulative += float(result.terminal_weights[index]) / selected_mass
            if threshold <= cumulative:
                selected_particle = index
                break

    selected_answer = canonical_answers[selected_particle]
    result.answer_text = result.particle_texts[selected_particle]
    result.answer_tokens = result.particle_tokens[selected_particle]
    result.diagnostics["smc_terminal_draw_particle"] = old_particle
    result.diagnostics["chosen_island"] = selected_particle // int(
        result.diagnostics["particles_per_island"]
    )
    result.diagnostics["chosen_particle"] = selected_particle % int(
        result.diagnostics["particles_per_island"]
    )
    readout_seconds = time.perf_counter() - readout_start
    result.diagnostics["answer_power_seconds"] = readout_seconds
    result.diagnostics["latency_seconds"] += readout_seconds

    answer_power = {
        "gamma": cfg.gamma,
        "selected_answer": selected_answer,
        "selected_particle": selected_particle,
        "answers": [
            {
                "answer": answer,
                "mass": answer_masses[answer],
                "probability": probability,
            }
            for answer, probability in zip(answers, answer_probabilities, strict=True)
        ],
        "seconds": readout_seconds,
    }

    evaluation = evaluate(cfg.dataset, example, result.answer_text)
    particle_evaluations = [evaluate(cfg.dataset, example, text) for text in result.particle_texts]
    result.diagnostics["coverage_at_32"] = any(item["correct"] for item in particle_evaluations)
    record = {
        "sample_id": example.uid,
        "category": example.category,
        "question": example.question,
        "references": example.answers,
        "answer": result.answer_text,
        "answer_tokens": result.answer_tokens,
        "evaluation": evaluation,
        "particle_answers": result.particle_texts,
        "particle_canonical_answers": canonical_answers,
        "particle_tokens": result.particle_tokens,
        "particle_evaluations": particle_evaluations,
        "terminal_weights": result.terminal_weights,
        "answer_power": answer_power,
        "diagnostics": result.diagnostics,
    }
    return record


def run(cfg: Config) -> Path:
    """Run the configured baseline or ReSight-SMC and save results incrementally."""

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    examples = load_examples(cfg)
    writer = ArtifactWriter(cfg)
    records: list[dict[str, Any]] = []
    answer_rng = random.Random(cfg.seed)

    if cfg.method in {"base", "cot"} and cfg.engine == "vllm":
        method = VLLMBaseline(cfg)
        progress = tqdm(total=len(examples), desc=f"{cfg.dataset}/{cfg.method}")
        for start in range(0, len(examples), cfg.baseline_batch_size):
            batch = examples[start : start + cfg.baseline_batch_size]
            questions = [format_question(example, cfg) for example in batch]
            results = method.run_batch([example.image for example in batch], questions)
            for example, result in zip(batch, results):
                record = _baseline_record(cfg, example, result)
                writer.prediction(record)
                records.append(record)
                writer.metrics(_metrics(records, writer.run_dir))
                progress.update(1)
        progress.close()
    else:
        if cfg.method in {"base", "cot"}:
            method = TransformersBaseline(cfg)
            smc = None
        else:
            runtime = TransformersVLM(cfg)
            generator = torch.Generator(device=runtime.device).manual_seed(cfg.seed)
            smc = PowerSMC(runtime, cfg, generator)
            method = None
        for example in tqdm(examples, desc=f"{cfg.dataset}/{cfg.method}"):
            question = format_question(example, cfg)
            if smc is None:
                assert method is not None
                record = _baseline_record(
                    cfg, example, method.run(example.image, question)
                )
            else:
                result = smc.run(example, question)
                record = _smc_record(cfg, example, result, answer_rng)
            writer.prediction(record)
            records.append(record)
            writer.metrics(_metrics(records, writer.run_dir))

    print(f"Run complete: {writer.run_dir.resolve()}")
    print(_metrics(records, writer.run_dir))
    return writer.run_dir
