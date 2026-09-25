from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .data import Example


def extract_answer(text: str) -> str:
    cleaned = text.strip()
    # TRACE-RL canonical responses end with {"answer": ...}.
    for line in reversed(cleaned.splitlines()):
        candidate = line.strip().strip("`")
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "answer" in value:
            return str(value["answer"]).strip()
    patterns = [r"final\s*answer\s*[:：]\s*(.+)", r"answer\s*[:：]\s*(.+)", r"答案\s*[:：]\s*(.+)"]
    for pattern in patterns:
        matches = list(re.finditer(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip().splitlines()[0].strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", cleaned)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[-1] if lines else cleaned


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\%", "%")
    value = value.strip(" \t\n\r.,;:!?，。；：！？'\"`[](){}")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s+(?:is\s+)?", "", value)
    return value.strip()


def choice_letter(text: str, choices: list[str] | None = None) -> str | None:
    answer = extract_answer(text)
    match = re.search(r"(?:^|[\s(\[])([A-Z])(?:$|[\s).,:：\]])", answer.upper())
    if match:
        return match.group(1)
    if choices:
        target = normalize(answer)
        for index, choice in enumerate(choices):
            if normalize(choice) == target:
                return chr(ord("A") + index)
    return None


def _choice_reference(example: Example) -> str | None:
    value = normalize(example.answers[0]).upper()
    if len(value) == 1 and value.isalpha():
        return value
    for index, choice in enumerate(example.choices):
        if normalize(choice) == normalize(example.answers[0]):
            return chr(ord("A") + index)
    return None


def _mathvista(example: Example, prediction: str) -> dict[str, Any]:
    parsed = extract_answer(prediction)
    answer_type = str(example.metadata.get("answer_type", "")).lower()
    question_type = str(example.metadata.get("question_type", "")).lower()
    reference = example.answers[0]
    normalized_reference = str(reference)
    normalized_prediction: str | None
    if question_type == "multi_choice" or example.choices:
        extraction = parsed.strip()
        parenthesized = re.findall(r"\(([a-zA-Z])\)", extraction)
        if parenthesized:
            extraction = parenthesized[0].upper()
        labels = [chr(ord("A") + i) for i in range(len(example.choices))]
        if extraction in labels:
            index = labels.index(extraction)
        else:
            distances = [_edit_distance(extraction, choice) for choice in example.choices]
            index = distances.index(min(distances)) if distances else None
        normalized_prediction = example.choices[index] if index is not None else None
        ref_letter = _choice_reference(example)
        if ref_letter and ord(ref_letter) - ord("A") < len(example.choices):
            normalized_reference = example.choices[ord(ref_letter) - ord("A")]
    elif answer_type in {"integer", "float"}:
        try:
            number = float(parsed)
            precision = 0 if answer_type == "integer" else int(example.metadata.get("precision", 0))
            normalized_prediction = (
                str(int(number)) if answer_type == "integer" else str(round(number, precision))
            )
            ref_number = float(normalized_reference)
            normalized_reference = (
                str(int(ref_number))
                if answer_type == "integer"
                else str(round(ref_number, precision))
            )
        except (ValueError, OverflowError):
            normalized_prediction = None
    elif answer_type == "list":
        normalized_prediction = str(parsed)
    else:
        normalized_prediction = normalize(parsed)
        normalized_reference = normalize(normalized_reference)
    correct = normalized_prediction is not None and normalized_prediction == normalized_reference
    return {
        "correct": correct,
        "score": float(correct),
        "parsed_answer": normalized_prediction if normalized_prediction is not None else parsed,
        "reference": reference,
        "metric": "mathvista_official_normalized_accuracy",
    }


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right))
            )
        previous = current
    return previous[-1]


def _logicvista(example: Example, prediction: str) -> dict[str, Any]:
    """Match single- and multi-choice labels exactly, ignoring label order."""

    parsed = extract_answer(prediction)
    predicted = sorted(
        set(re.findall(r"(?<![A-Z0-9])(?:[A-G]|[1-9])(?![A-Z0-9])", parsed.upper()))
    )
    reference = sorted(part.strip().upper() for part in example.answers[0].split(","))
    correct = predicted == reference
    return {
        "correct": correct,
        "score": float(correct),
        "parsed_answer": ", ".join(predicted),
        "reference": ", ".join(reference),
        "metric": "logicvista_exact_choice_set_accuracy",
    }


def canonical_answer(dataset: str, example: Example, prediction: str) -> str:
    """Map particle text to answer space using question type, not the reference answer."""

    parsed = extract_answer(prediction)
    if dataset == "mathvista":
        answer_type = str(example.metadata.get("answer_type", "")).lower()
        question_type = str(example.metadata.get("question_type", "")).lower()
        if question_type == "multi_choice" or example.choices:
            extraction = parsed.strip()
            parenthesized = re.findall(r"\(([a-zA-Z])\)", extraction)
            if parenthesized:
                extraction = parenthesized[0].upper()
            labels = [chr(ord("A") + i) for i in range(len(example.choices))]
            if extraction in labels:
                return normalize(example.choices[labels.index(extraction)])
            if example.choices:
                distances = [_edit_distance(extraction, choice) for choice in example.choices]
                return normalize(example.choices[distances.index(min(distances))])
        elif answer_type in {"integer", "float"}:
            try:
                number = float(parsed)
                precision = 0 if answer_type == "integer" else int(
                    example.metadata.get("precision", 0)
                )
                return str(int(number)) if answer_type == "integer" else str(round(number, precision))
            except (ValueError, OverflowError):
                pass
        return normalize(parsed)

    if dataset == "logicvista":
        labels = sorted(
            set(re.findall(r"(?<![A-Z0-9])(?:[A-G]|[1-9])(?![A-Z0-9])", parsed.upper()))
        )
        return ", ".join(labels) if labels else normalize(parsed)

    if dataset.startswith("mmstar") or dataset == "realworldqa":
        letter = choice_letter(prediction, example.choices)
        if letter is not None:
            return letter

    return normalize(parsed)


def evaluate(dataset: str, example: Example, prediction: str) -> dict[str, Any]:
    if dataset == "mathvista":
        return _mathvista(example, prediction)
    if dataset == "logicvista":
        return _logicvista(example, prediction)
    if dataset == "realworldqa":
        reference = example.answers[0].strip()
        if reference.upper() in {"A", "B", "C", "D"}:
            parsed = choice_letter(prediction)
            normalized_reference = reference.upper()
        else:
            parsed = extract_answer(prediction).strip().rstrip(".").lower()
            normalized_reference = reference.lower()
        correct = parsed is not None and parsed == normalized_reference
        return {
            "correct": correct,
            "score": float(correct),
            "parsed_answer": parsed,
            "reference": normalized_reference,
            "metric": "realworldqa_exact_match",
        }
    if dataset.startswith("mmstar"):
        pred = choice_letter(prediction, example.choices)
        reference = _choice_reference(example)
        correct = pred is not None and pred == reference
        return {
            "correct": correct,
            "score": float(correct),
            "parsed_answer": pred,
            "reference": reference,
            "metric": "multiple_choice_accuracy",
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["evaluation"]["score"]) for record in records]
    categories: dict[str, list[float]] = {}
    for record, score in zip(records, scores):
        categories.setdefault(record.get("category") or "unknown", []).append(score)
    metrics = {
        "num_examples": len(records),
        "accuracy": sum(scores) / len(scores) if scores else None,
        "num_correct": sum(bool(record["evaluation"]["correct"]) for record in records),
        "by_category": {
            key: {"num_examples": len(values), "accuracy": sum(values) / len(values)}
            for key, values in sorted(categories.items())
        },
    }
    return metrics
