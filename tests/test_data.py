from PIL import Image

from resight_smc.config import Config
from resight_smc.data import Example, format_question


def test_other_datasets_keep_the_generic_prompt_suffix():
    example = Example("1", "Question?", Image.new("RGB", (4, 4)), ["A"])
    cfg = Config(dataset="mathvista", prompt_mode="direct")
    assert format_question(example, cfg) == f"Question?\n\n{cfg.direct_suffix}"


def test_realworldqa_keeps_released_direct_instruction():
    question = "Question? Please answer directly with a single word or number."
    example = Example("1", question, Image.new("RGB", (4, 4)), ["Yes"])
    direct = format_question(example, Config(dataset="realworldqa", prompt_mode="direct"))
    assert direct == question
