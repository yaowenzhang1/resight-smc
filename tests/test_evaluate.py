import random

from PIL import Image

from resight_smc.config import Config
from resight_smc.data import Example
from resight_smc.evaluate import canonical_answer, evaluate
from resight_smc.runner import _smc_record
from resight_smc.smc import SMCResult


def test_multiple_choice_and_mathvista_integer():
    choice = Example("1", "q", Image.new("RGB", (4, 4)), ["B"])
    assert evaluate("mmstar_p", choice, "Final answer: B")["correct"]

    integer = Example(
        "2",
        "q",
        Image.new("RGB", (4, 4)),
        ["12"],
        metadata={"answer_type": "integer", "question_type": "free_form"},
    )
    assert evaluate("mathvista", integer, "Final answer: 12")["correct"]
    assert not evaluate("mathvista", integer, "Final answer: infinity")["correct"]


def test_realworldqa_multiple_choice_and_short_answer():
    choice = Example("1", "q", Image.new("RGB", (4, 4)), ["B"])
    assert evaluate("realworldqa", choice, "Final answer: B.")["correct"]

    short = Example("2", "q", Image.new("RGB", (4, 4)), ["Downhill"])
    assert evaluate("realworldqa", short, "Final answer: downhill.")["correct"]


def test_trace_json_answer_is_extracted():
    choice = Example("1", "q", Image.new("RGB", (4, 4)), ["C"])
    assert evaluate("realworldqa", choice, 'reasoning\n{"answer": "C"}')["correct"]


def test_logicvista_single_multi_and_numeric_choices():
    image = Image.new("RGB", (4, 4))
    single = Example("1", "q", image, ["C"])
    multi = Example("2", "q", image, ["A, C"])
    numeric = Example("3", "q", image, ["3"])

    assert evaluate("logicvista", single, "Final answer: C.")["correct"]
    assert evaluate("logicvista", multi, "Final answer: C and A.")["correct"]
    assert not evaluate("logicvista", multi, "Final answer: A, C, D.")["correct"]
    assert evaluate("logicvista", numeric, "Final answer: 3.")["correct"]


def test_answer_power_uses_weighted_answer_mass_without_reference_leakage():
    image = Image.new("RGB", (4, 4))
    particle_texts = [
        "Final answer: B",
        "Final answer: B",
        "Final answer: A",
        "Final answer: A",
    ]

    def record(reference: str):
        cfg = Config(
            dataset="mmstar_p",
            method="island_smc",
            gamma=2.0,
            num_particles=4,
            num_islands=2,
        )
        cfg.finalize()
        result = SMCResult(
            answer_text=particle_texts[0],
            answer_tokens=[0],
            particle_texts=particle_texts,
            particle_tokens=[[0], [1], [2], [3]],
            terminal_weights=[0.1, 0.2, 0.3, 0.4],
            diagnostics={
                "latency_seconds": 1.0,
                "peak_vram_gb": 0.0,
                "chosen_island": 0,
                "chosen_particle": 0,
                "particles_per_island": 2,
            },
        )
        example = Example("x", "q", image, [reference])
        return _smc_record(cfg, example, result, random.Random(7))

    left = record("A")
    right = record("B")
    assert left["answer_power"]["answers"] == right["answer_power"]["answers"]
    assert left["answer_power"]["selected_particle"] == right["answer_power"]["selected_particle"]
    probabilities = {
        item["answer"]: item["probability"] for item in left["answer_power"]["answers"]
    }
    assert abs(probabilities["A"] - 0.49 / 0.58) < 1e-12
    assert abs(probabilities["B"] - 0.09 / 0.58) < 1e-12


def test_mathvista_canonical_answer_uses_type_not_reference():
    image = Image.new("RGB", (4, 4))
    first = Example(
        "1",
        "q",
        image,
        ["12"],
        metadata={"answer_type": "integer", "question_type": "free_form"},
    )
    second = Example(
        "2",
        "q",
        image,
        ["999"],
        metadata={"answer_type": "integer", "question_type": "free_form"},
    )
    assert canonical_answer("mathvista", first, "Final answer: 12.0") == "12"
    assert canonical_answer("mathvista", second, "Final answer: 12.0") == "12"
