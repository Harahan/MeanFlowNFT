"""Wan2.1 MeanFlowNFT training entry point."""

import argparse
import yaml

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.trainers.wan_meanflow_nft_trainer import (
    WanMeanFlowNFTTrainer,
)


TRAINER_REGISTRY = {
    "meanflow_nft": WanMeanFlowNFTTrainer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.1 MeanFlowNFT Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--trainer",
        type=str,
        default="meanflow_nft",
        choices=list(TRAINER_REGISTRY.keys()),
        help="Trainer type (only meanflow_nft is available on this branch)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g. train.seed=123",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load train.resume_from and run test-set/VBench evaluation once.",
    )
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI overrides such as ``train.seed=123`` into a dictionary."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}. Expected key=value.")
        key, raw_value = item.split("=", 1)
        value = "" if raw_value == "" else yaml.safe_load(raw_value)
        overrides[key] = value
    return overrides


def main():
    args = parse_args()

    # Load config with optional CLI overrides
    overrides = parse_overrides(args.override) if args.override else None
    config = MeanFlowNFTConfig.from_yaml(args.config, overrides=overrides)

    # Dispatch to the appropriate trainer
    trainer_cls = TRAINER_REGISTRY[args.trainer]
    trainer = trainer_cls(config)
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
