"""MeanFlowNFT training entry point.

Supported workflows:
- ``anyflow_pretrain``: Stage 1 forward training.
- ``anyflow_onpolicy``: Stage 2 on-policy distribution matching.
- ``meanflow_nft``: MeanFlowNFT reinforcement learning.
"""

# Pre-import pkg_resources BEFORE the heavy trainer stack (torch / transformers
# / diffusers / ...). Some of those imports interfere with a later
# ``import pkg_resources`` (ImageReward imports it internally), which surfaces
# as a spurious ``ModuleNotFoundError: No module named 'pkg_resources'`` at
# eval-time reward loading even though setuptools/pkg_resources ARE installed.
# Importing it first caches the real module in ``sys.modules`` so the later
# import hits the cache and bypasses any finder/hook interference.
try:
    import pkg_resources  # type: ignore  # noqa: F401
except Exception:
    pass

import argparse
import yaml

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.trainers.meanflow_nft_trainer import MeanFlowNFTTrainer
from meanflownft.trainers.anyflow_onpolicy_trainer import AnyFlowOnPolicyTrainer
from meanflownft.trainers.anyflow_pretrain_trainer import AnyFlowPretrainTrainer


TRAINER_REGISTRY = {
    "anyflow_pretrain": AnyFlowPretrainTrainer,
    "anyflow_onpolicy": AnyFlowOnPolicyTrainer,
    "meanflow_nft": MeanFlowNFTTrainer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MeanFlowNFT Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--trainer",
        type=str,
        default="anyflow_pretrain",
        choices=list(TRAINER_REGISTRY.keys()),
        help="Trainer type to use (default: anyflow_pretrain)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key=value format, e.g. train.seed=123",
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
    trainer.train()


if __name__ == "__main__":
    main()
