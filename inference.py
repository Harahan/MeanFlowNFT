"""Command-line entry point for SD3.5-Medium / MeanFlowNFT inference."""

from __future__ import annotations

import argparse
import logging

import yaml

from meanflownft.inference import MeanFlowNFTInference, InferenceConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean SD3.5 or an ordered 1/2/3-stage MeanFlowNFT prefix."
        )
    )
    parser.add_argument("config", help="Path to sd35m_meanflow_nft.yaml")
    parser.add_argument(
        "--override",
        nargs="+",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Scalar YAML overrides; may be repeated. Example: "
            "--override num_steps=8 output_dir=./outputs/steps_8"
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        dest="cli_prompts",
        help="Generate this prompt; repeat for multiple prompts.",
    )
    parser.add_argument(
        "--prompt_file",
        default="",
        help="Use a .txt/.json/.jsonl prompt file instead of the config source.",
    )
    return parser.parse_args()


def parse_overrides(groups: list[list[str]]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for group in groups:
        for item in group:
            if "=" not in item:
                raise ValueError(
                    f"Invalid override {item!r}; expected KEY=VALUE"
                )
            key, raw_value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid empty override key in {item!r}")
            overrides[key] = (
                "" if raw_value == "" else yaml.safe_load(raw_value)
            )
    return overrides


def main() -> None:
    args = parse_args()
    config = InferenceConfig.from_yaml(
        args.config,
        overrides=parse_overrides(args.override),
    )

    # Explicit CLI prompt sources take precedence over YAML defaults.
    if args.cli_prompts:
        config.prompts = args.cli_prompts
    elif args.prompt_file:
        config.prompts = []
        config.prompt_file = args.prompt_file

    config.validate()
    engine = MeanFlowNFTInference(config)
    try:
        engine.setup()
        engine.run()
    except BaseException:
        engine.close(synchronize=False)
        logger.exception("SD3.5 / MeanFlowNFT inference failed")
        raise


if __name__ == "__main__":
    main()
