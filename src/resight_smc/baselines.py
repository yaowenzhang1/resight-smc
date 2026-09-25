from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image

from .config import Config
from .model import TransformersVLM, prepare_processor_compatibility


@dataclass
class BaselineResult:
    text: str
    tokens: list[int]
    latency_seconds: float
    peak_vram_gb: float


class TransformersBaseline:
    """Run standard LVLM generation without visual decoding enhancements."""

    def __init__(self, cfg: Config):
        self.runtime = TransformersVLM(cfg)
        self.cfg = cfg

    def run(self, image: Image.Image, question: str) -> BaselineResult:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        prepared = self.runtime.prepare_example(image, question)
        generate_kwargs = {
            "input_ids": prepared.input_ids,
            "attention_mask": prepared.attention_mask,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.temperature > 0,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "use_cache": True,
            **prepared.model_kwargs,
        }
        with torch.inference_mode():
            output = self.runtime.model.generate(**generate_kwargs)
        tokens = output[0, prepared.input_ids.shape[1] :].tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        return BaselineResult(self.runtime.decode(tokens), tokens, latency, peak)


class VLLMBaseline:
    """Use vLLM for base, CoT, and RL baselines without changing sampling."""

    def __init__(self, cfg: Config):
        from transformers import AutoProcessor

        self.cfg = cfg
        use_fast = prepare_processor_compatibility(cfg.model)
        if use_fast:
            # A separate vLLM engine process would reimport Transformers and miss
            # the compatibility fix above. Single-process mode retains batching
            # and KV caching for these single-GPU offline runs.
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(
            cfg.model,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
            **({"use_fast": True} if use_fast else {}),
        )
        self.llm = LLM(
            model=cfg.model,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
            mm_processor_kwargs={
                "min_pixels": cfg.min_pixels,
                "max_pixels": cfg.max_pixels,
            },
        )
        self.params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )

    def _prompt(self, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            },
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def run_batch(
        self, images: Sequence[Image.Image], questions: Sequence[str]
    ) -> list[BaselineResult]:
        requests = [
            {
                "prompt": self._prompt(question),
                "multi_modal_data": {"image": image.convert("RGB")},
            }
            for image, question in zip(images, questions)
        ]
        start = time.perf_counter()
        outputs = self.llm.generate(requests, self.params)
        elapsed = time.perf_counter() - start
        per_example = elapsed / len(outputs)
        return [
            BaselineResult(
                text=output.outputs[0].text,
                tokens=list(output.outputs[0].token_ids),
                latency_seconds=per_example,
                peak_vram_gb=0.0,
            )
            for output in outputs
        ]
