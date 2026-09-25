from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one model or dataset")
    parser.add_argument("--model")
    parser.add_argument(
        "--dataset",
        choices=[
            "mathvista",
            "mmstar_r",
            "mmstar_p",
            "logicvista",
            "realworldqa",
        ],
    )
    args = parser.parse_args()
    if args.model:
        from huggingface_hub import snapshot_download

        path = snapshot_download(args.model)
        print(path)
    if args.dataset:
        from .config import Config
        from .data import load_examples

        cfg = Config(dataset=args.dataset, max_samples=None)
        cfg.finalize()
        examples = load_examples(cfg)
        print(f"{args.dataset}: {len(examples)} examples")
