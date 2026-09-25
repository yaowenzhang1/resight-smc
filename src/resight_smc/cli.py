from __future__ import annotations

from .config import load_config
from .runner import run


def main() -> None:
    run(load_config())
