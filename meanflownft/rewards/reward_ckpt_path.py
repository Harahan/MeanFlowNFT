"""
Default reward checkpoint path configuration.

Resolution priority:
1) MEANFLOWNFT_REWARD_CKPT_PATH environment variable
2) <repository>/models/reward_ckpts

Can also be overridden at runtime via EvalConfig.reward_ckpt_path.
"""

import os
from pathlib import Path


def _resolve_default_ckpt_path() -> str:
    env = os.environ.get("MEANFLOWNFT_REWARD_CKPT_PATH", "").strip()
    if env:
        return os.path.expanduser(env)

    repository_root = Path(__file__).resolve().parents[2]
    return str(repository_root / "models" / "reward_ckpts")


CKPT_PATH = _resolve_default_ckpt_path()


def set_ckpt_path(path: str) -> None:
    """Override the global reward checkpoint path."""
    global CKPT_PATH
    if path:
        CKPT_PATH = os.path.expanduser(path)
