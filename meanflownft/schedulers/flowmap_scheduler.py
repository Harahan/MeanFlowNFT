"""
Flow Map Scheduler for AnyFlow distillation.

Implements the scheduler used by AnyFlow (arXiv 2605.13724). Unlike an
ordinary Euler scheduler, which advances one local step using an instantaneous
velocity prediction,
``FlowMapScheduler.step`` accepts an arbitrary ``(timestep, r_timestep)``
pair and applies the closed-form flow-map jump::

    prev_sample = sample - (timestep - r_timestep) / num_train_timesteps * model_output

where ``model_output`` is interpreted as the *average* velocity of the flow
map over the interval ``[r, t]``. When ``r -> t``, this reduces to the
standard instantaneous Euler step; when ``r = 0``, this maps directly from
``z_t`` to ``z_0`` (the consistency endpoint).

Aligned with AnyFlow ``far/schedulers/scheduling_flowmap_euler_discrete.py``.

Key design notes:
- ``set_timesteps(N)`` produces ``N + 1`` timesteps via ``linspace(1.0, 0.0, N+1)``,
  with optional static shift transform applied. The trailing ``0`` is the
  endpoint timestep used for shortcut rollout (the final segment ends at 0).
- ``apply_shift`` mirrors the SD3 static-shift formula and is the same one
  used by :func:`meanflownft.models.sd35.compute_sigmas_sd35`.
- ``get_train_weight(t)`` returns timestep-dependent loss weights for the
  forward training stage (gaussian / beta08 / uniform). Aligned with AnyFlow
  exactly.
- ``scale_noise(sample, t, noise)`` performs the forward diffusion mixing
  used by both pretrain and on-policy stages: ``z_t = (1 - t/T) * x + (t/T) * noise``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowMapScheduler(SchedulerMixin, ConfigMixin):
    """Flow map scheduler with arbitrary ``(t, r)`` step.

    Aligned with AnyFlow ``FlowMapDiscreteScheduler``: same step formula,
    same shift transform, same train-weight options.

    Args:
        num_train_timesteps: Total training timesteps (defines the [0, T]
            grid; sigmas in the rest of MeanFlowNFT are normalized by this value).
        shift: Static SD3-style shift parameter applied to ``set_timesteps``
            output. ``shift = 1.0`` means no shift; ``shift = 3.0`` matches
            SD3.5 default.
        weight_type: Loss-weight curve over t for forward training:
            'gaussian' (peaked at t = T/2) / 'beta08' (t * (1-t)^0.5) / 'uniform'.
    """

    # Per diffusers convention: lower order means simpler (Euler-style).
    order = 1
    _compatibles: list[str] = []

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        weight_type: str = "gaussian",
    ):
        self.set_timesteps(num_train_timesteps, device="cpu")
        self.set_train_weight(weight_type)

    # ------------------------------------------------------------------
    # SD3 static shift:
    # sigma' = alpha * sigma / (1 + (alpha - 1) * sigma)
    # ------------------------------------------------------------------

    def apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Apply the configured SD3 shift to sigmas in ``[0, 1]``."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (
            1.0 + (self.config.shift - 1.0) * sigmas
        )

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        """Build a length-``(N+1)`` timestep grid in raw model timestep units.

        Aligned with AnyFlow: ``timesteps = apply_shift(linspace(1, 0, N+1)) * T``.
        The trailing 0 is included so that shortcut rollout can take the final
        ``(timesteps[N-1], 0)`` jump.

        """
        if num_inference_steps is None or num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )
        timesteps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device,
        )
        timesteps = self.apply_shift(timesteps)
        self.timesteps = timesteps * self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ------------------------------------------------------------------
    # Forward diffusion mixing (used by both pretrain and on-policy)
    # ------------------------------------------------------------------

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Linear forward-noising: ``z_t = (1 - t/T) * x + (t/T) * noise``.

        Aligned with AnyFlow ``FlowMapDiscreteScheduler.scale_noise``.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device, dtype=sample.dtype)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        return timestep * noise + (1.0 - timestep) * sample

    # ------------------------------------------------------------------
    # Flow-map step: prev = sample - (t - r) / T * model_output
    # ------------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        r_timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """One flow-map jump from ``z_t`` to ``z_r``.

        ``model_output`` is interpreted as the average velocity over the
        interval ``[r, t]`` (a "flow map"). When ``r -> t`` this reduces to
        an instantaneous Euler step; when ``r = 0`` this is the endpoint
        consistency mapping ``z_t -> z_0``.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=sample.device, dtype=model_output.dtype)
        if not torch.is_tensor(r_timestep):
            r_timestep = torch.tensor(r_timestep, device=sample.device, dtype=model_output.dtype)
        timestep = timestep.to(device=model_output.device) / self.config.num_train_timesteps
        r_timestep = r_timestep.to(device=model_output.device) / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output
        return prev_sample.to(model_output.dtype)

    # ------------------------------------------------------------------
    # Loss weighting over t (for forward training stage)
    # ------------------------------------------------------------------

    def set_train_weight(self, weight_type: str) -> None:
        """Precompute per-timestep loss weights of length ``num_train_timesteps + 1``.

        Aligned with AnyFlow ``set_train_weight`` (gaussian / beta08 / uniform).
        Each curve is normalized so that ``sum(weights) == num_train_timesteps``,
        so changing the weight type does not rescale the loss magnitude.
        """
        T = self.config.num_train_timesteps
        # Keep an immutable training grid. ``set_timesteps`` is also used by
        # rollout and may replace ``self.timesteps`` with a short inference
        # grid; loss lookup must remain on all T+1 training points.
        raw = torch.linspace(1.0, 0.0, T + 1, dtype=torch.float64)
        self.train_timesteps = self.apply_shift(raw) * T
        x = self.train_timesteps
        if weight_type == "gaussian":
            y = torch.exp(-2.0 * ((x - T / 2.0) / T) ** 2)
            y_shifted = y - y.min()
            weights = y_shifted * (T / y_shifted.sum())
        elif weight_type == "beta08":
            t = x / T
            y = (t ** 1.0) * ((1.0 - t) ** 0.5)
            weights = y * (T / y.sum())
        elif weight_type == "uniform":
            weights = torch.ones_like(x) * (T / x.numel())
        else:
            raise ValueError(
                f"Invalid weight_type {weight_type!r}; must be one of "
                f"'gaussian', 'beta08', 'uniform'."
            )
        # Cast to float32 for stable lookups and broadcast.
        self.linear_timesteps_weights = weights.to(dtype=torch.float32)

    @torch.no_grad()
    def get_train_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Look up per-sample weight for a tensor of (raw) timesteps.

        Aligned with AnyFlow's nearest-neighbor matching (no interpolation).
        """
        ref = self.train_timesteps.to(timesteps.device)
        # (num_train_timesteps + 1, num_query)
        diff = (ref.unsqueeze(1) - timesteps.flatten().unsqueeze(0).to(ref.dtype)).abs()
        idx = torch.argmin(diff, dim=0).reshape(timesteps.shape)
        return self.linear_timesteps_weights.to(timesteps.device)[idx]
