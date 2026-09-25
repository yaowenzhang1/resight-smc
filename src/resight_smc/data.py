from __future__ import annotations

import ast
import io
import json
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .config import Config


@dataclass
class Example:
    uid: str
    question: str
    image: Image.Image
    answers: list[str]
    choices: list[str] = field(default_factory=list)
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


DATASETS = {
    "mathvista": ("AI4Math/MathVista", "testmini"),
    "mmstar_r": ("Lin-Chen/MMStar", "val"),
    "mmstar_p": ("Lin-Chen/MMStar", "val"),
    "logicvista": ("lscpku/LogicVista", "test"),
    "realworldqa": ("xai-org/RealworldQA", "test"),
}

MMSTAR_REASONING = {
    "instance reasoning",
    "logical reasoning",
    "math",
    "science & technology",
}
MMSTAR_PERCEPTION = {"coarse perception", "fine-grained perception"}
def _image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    return Image.open(value).convert("RGB")


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and value.strip().startswith("["):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return [value]


def _message_text(value: Any) -> str:
    messages = _sequence(value)
    if not messages:
        return ""
    message = next(
        (item for item in messages if isinstance(item, dict) and item.get("role") == "user"),
        messages[0],
    )
    if not isinstance(message, dict):
        return str(message)
    content = message.get("content") or message.get("question") or ""
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("value"))
            for part in content
            if isinstance(part, dict) and (part.get("text") or part.get("value"))
        )
    return str(content)


def _hf_examples(cfg: Config) -> list[Example]:
    from datasets import load_dataset

    repo, default_split = DATASETS[cfg.dataset]
    split = cfg.split or default_split
    rows = load_dataset(repo, split=split)
    examples: list[Example] = []
    for index, item in enumerate(rows):
        row = dict(item)
        if cfg.dataset == "mathvista":
            raw_answer = row.get("answer")
            answers = [str(raw_answer)]
            example = Example(
                uid=str(row.get("pid") or row.get("id") or index),
                question=str(row.get("query") or row.get("question")),
                image=_image(row.get("decoded_image") or row.get("image")),
                answers=answers,
                choices=[str(x) for x in (row.get("choices") or [])],
                category=str(row.get("task") or row.get("question_type") or ""),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"image", "decoded_image"}
                },
            )
        elif cfg.dataset.startswith("mmstar"):
            category = str(row.get("category") or "")
            wanted = MMSTAR_REASONING if cfg.dataset == "mmstar_r" else MMSTAR_PERCEPTION
            if category.lower() not in wanted:
                continue
            example = Example(
                uid=str(row.get("id") or index),
                question=_message_text(row.get("messages") or row.get("question")),
                image=_image(row.get("image")),
                answers=[str(row.get("answer"))],
                category=category,
                metadata={key: value for key, value in row.items() if key != "image"},
            )
        elif cfg.dataset == "realworldqa":
            answer = str(row.get("answer"))
            example = Example(
                uid=str(row.get("id") or row.get("image_path") or index),
                question=str(row.get("question")),
                image=_image(row.get("image")),
                answers=[answer],
                category=(
                    "multiple_choice"
                    if answer.strip().upper() in {"A", "B", "C", "D"}
                    else "short_answer"
                ),
                metadata={key: value for key, value in row.items() if key != "image"},
            )
        elif cfg.dataset == "logicvista":
            skills = [str(value) for value in _sequence(row.get("skill"))]
            example = Example(
                uid=str(row.get("id") or index),
                question=str(row.get("question") or ""),
                image=_image(row.get("image")),
                answers=[str(row.get("answer") or "")],
                category=", ".join(skills),
                metadata={key: value for key, value in row.items() if key != "image"},
            )
        examples.append(example)
    return examples


def load_examples(cfg: Config) -> list[Example]:
    if cfg.dataset in DATASETS:
        examples = _hf_examples(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    start = cfg.start_index
    end = None if cfg.max_samples is None else start + cfg.max_samples
    return examples[start:end]


def format_question(example: Example, cfg: Config) -> str:
    if cfg.dataset == "realworldqa" and cfg.prompt_mode == "direct":
        # Released RealWorldQA questions already include answer instructions.
        return example.question.strip()
    suffix = cfg.cot_suffix if cfg.prompt_mode == "cot" else cfg.direct_suffix
    return f"{example.question.strip()}\n\n{suffix}"
