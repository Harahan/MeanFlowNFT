"""
Fast model initialization utilities for MeanFlowNFT.

Avoids redundant weight initialization when loading pretrained models.
Instead of: allocate on CPU → random init → overwrite with pretrained weights,
this does: allocate on meta device → materialize empty → overwrite with pretrained weights.

Benefits:
- ~2x faster model loading (skip kaiming_uniform_ etc.)
- No RNG consumption during model construction (determinism-safe)
- Lower peak memory (no temporary random tensors)

Migrated from DiffusionNFT/flow_grpo/fast_init.py.

Usage::

    from meanflownft.utils.fast_init import fast_init

    with fast_init(torch.device("cpu")):
        model = SomeModel.from_pretrained("path/to/model")
    # Weights are loaded by from_pretrained, so no random init needed.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn


# Original __init__ methods that we monkey-patch during fast_init.
_ORIGINAL_INITS: dict[type[nn.Module], Any] = {
    nn.Linear: nn.Linear.__init__,
    nn.Embedding: nn.Embedding.__init__,
    nn.LayerNorm: nn.LayerNorm.__init__,
}


def _get_fast_init(cls: type[nn.Module], device: torch.device):
    """Create a patched __init__ that constructs on meta device then materializes."""
    assert cls in _ORIGINAL_INITS

    def _fast_init(self, *args, **kwargs):
        kwargs.pop("device", None)
        _ORIGINAL_INITS[cls](self, *args, **kwargs, device="meta")
        self.to_empty(device=device)

    return _fast_init


@contextlib.contextmanager
def no_init_weights():
    """Disable all weight initialization ops inside the context.

    Patches nn.init.* functions (kaiming_uniform_, normal_, zeros_, ones_,
    uniform_, xavier_uniform_, xavier_normal_, constant_) to be no-ops.
    This prevents any RNG consumption from weight initialization.
    """
    init_funcs = [
        "kaiming_uniform_",
        "kaiming_normal_",
        "zeros_",
        "ones_",
        "normal_",
        "uniform_",
        "xavier_uniform_",
        "xavier_normal_",
        "constant_",
    ]
    originals = {name: getattr(nn.init, name) for name in init_funcs if hasattr(nn.init, name)}

    for name in originals:
        setattr(nn.init, name, lambda *a, **kw: None)

    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(nn.init, name, orig)


@contextlib.contextmanager
def fast_init(device: torch.device, init_weights: bool = False):
    """Fast model construction: meta device + skip weight init.

    Monkey-patches nn.Linear, nn.Embedding, nn.LayerNorm to construct on
    the meta device (no memory allocation), then materialize as empty tensors
    on the target device. Combined with no_init_weights(), this skips all
    random initialization entirely.

    After the context exits, call model.load_state_dict() or
    from_pretrained() to fill in the actual weights.

    Args:
        device: Target device for materialized tensors (usually torch.device("cpu")).
        init_weights: If True, keep normal weight initialization (only use
            meta-device construction). If False (default), also disable
            nn.init.* functions for maximum speed and zero RNG consumption.

    Usage::

        with fast_init(torch.device("cpu")):
            model = AutoModel.from_pretrained("model_name")
        # model is now on CPU with pretrained weights loaded.
    """
    # Patch __init__ methods to use meta device
    for cls in _ORIGINAL_INITS:
        cls.__init__ = _get_fast_init(cls, device)

    ctx = contextlib.nullcontext() if init_weights else no_init_weights()
    try:
        with ctx:
            yield
    finally:
        for cls in _ORIGINAL_INITS:
            cls.__init__ = _ORIGINAL_INITS[cls]
