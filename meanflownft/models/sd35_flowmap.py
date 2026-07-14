"""
Flow-map (two-time) wrapper for the SD3.5 transformer.

This module turns a vanilla SD3.5 ``SD3Transformer2DModel`` (whose
``time_text_embed`` only conditions on a single timestep ``t``) into a
flow-map model conditioned on ``(t, r)`` per the AnyFlow paper
(https://arxiv.org/abs/2605.13724).

The wrapping replaces ``transformer.time_text_embed`` (an instance of
:class:`diffusers.models.embeddings.CombinedTimestepTextProjEmbeddings`)
in-place with :class:`SD3FlowMapTimeTextEmbed`, a drop-in replacement that:

1. Reuses the original ``time_proj``, ``timestep_embedder`` and
   ``text_embedder`` modules (so existing pretrained weights load as-is).
2. Adds a new ``delta_embedder`` (deep-copy of the original
   ``timestep_embedder`` so it inherits the same well-conditioned init).
3. Mixes the two embeddings via a fixed gate buffer:
   ``rt_emb = (1 - gate) * timestep_embedder(t) + gate * delta_embedder(delta_t)``,
   where ``delta_t`` is either ``r`` or ``t - r`` depending on
   ``deltatime_type``.

Critically, ``transformer.forward`` itself is **not modified** — the SD3
transformer always calls ``self.time_text_embed(timestep, pooled_projections)``
internally. To pass ``r_timestep`` from the trainer down into the wrapper,
we use a per-instance attribute ``_r_timestep_pending`` that the trainer
sets before each forward (via the :func:`with_r_timestep` context manager)
and the wrapper reads inside its forward. This is single-thread safe (each
distributed rank runs forward serially) and avoids monkey-patching
``transformer.forward``, keeping the wrapper compatible with both DDP and
FSDP wrap.

The gate is registered as a non-persistent buffer (NOT a learnable
parameter), aligned with the AnyFlow reference codebase. r-dependence is
learned through the trainable ``delta_embedder`` weights.

Usage::

    from meanflownft.models.sd35_flowmap import setup_flowmap_for_sd3, with_r_timestep

    # Once at model setup (BEFORE FSDP wrap, so the new module gets sharded):
    setup_flowmap_for_sd3(transformer, gate_value=0.25, deltatime_type="r")

    # In training step:
    with with_r_timestep(transformer, r_timestep_tensor):
        v = transformer(hidden_states=xt, timestep=t, pooled_projections=pooled,
                        encoder_hidden_states=prompt_embeds, return_dict=False)[0]
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drop-in wrapper for time_text_embed
# ---------------------------------------------------------------------------

class SD3FlowMapTimeTextEmbed(nn.Module):
    """Drop-in replacement for ``CombinedTimestepTextProjEmbeddings`` that
    conditions on both ``t`` and ``r`` (or ``t - r``).

    Constructed by deep-copying the ``timestep_embedder`` from the original
    module to initialize ``delta_embedder``; everything else (``time_proj``,
    ``text_embedder``) is reused by reference. This means the wrapped module
    behaves *identically* to the original at gate=0 (since ``delta_embedder``
    output is multiplied by 0).

    See module docstring for the full integration pattern.
    """

    def __init__(
        self,
        original: CombinedTimestepTextProjEmbeddings,
        gate_value: float = 0.25,
        deltatime_type: str = "r",
    ):
        super().__init__()
        if deltatime_type not in ("r", "t-r"):
            raise ValueError(
                f"deltatime_type must be 'r' or 't-r', got {deltatime_type!r}."
            )
        # Reuse existing modules in-place. ``time_proj`` is parameter-free and
        # ``text_embedder`` / ``timestep_embedder`` carry pretrained weights we
        # want to keep.
        self.time_proj = original.time_proj
        self.timestep_embedder = original.timestep_embedder
        self.text_embedder = original.text_embedder
        # Deep-copy the timestep_embedder weights to initialize the new
        # delta_embedder. After this point, ``delta_embedder`` is a separate
        # trainable module (its parameters are independent from
        # ``timestep_embedder``).
        self.delta_embedder = copy.deepcopy(original.timestep_embedder)
        # Force-unfreeze: when LoRA is active, setup_lora() has already frozen
        # the original timestep_embedder; deepcopy inherits that frozen state.
        # Without this explicit unfreeze, delta_embedder stays frozen and the
        # AnyFlow paper's core r-conditioning never trains (the gate=0.25 mix
        # would degenerate to (1-0.25)*timestep_emb + 0.25*dead_delta_emb).
        # AnyFlow Wan keeps delta_embedder trainable by including it in its
        # LoRA target_module_name list; we instead full-FT this small MLP
        # (~1.2M params for SD3.5) for simplicity.
        for p in self.delta_embedder.parameters():
            p.requires_grad = True
        # Fixed (non-learned) mixing ratio. Aligned with AnyFlow:
        # ``register_buffer`` with persistent=False means this is NOT saved to
        # state_dict and NOT a Parameter (so it stays out of the optimizer).
        self.register_buffer(
            "delta_emb_gate",
            torch.tensor([float(gate_value)], dtype=torch.float32),
            persistent=False,
        )
        self.deltatime_type = deltatime_type
        # Slot for per-call r_timestep injection. The trainer sets this via
        # the with_r_timestep context manager before each forward, and clears
        # it after. When None (e.g. pre-wrapping inference), the wrapper falls
        # back to r = t (degenerates to standard flow matching, identical to
        # the original module up to the additive delta_embedder term).
        self._r_timestep_pending: Optional[torch.Tensor] = None

    def forward(
        self, timestep: torch.Tensor, pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Same signature as ``CombinedTimestepTextProjEmbeddings.forward``.

        Reads ``r_timestep`` from ``self._r_timestep_pending``. When the slot
        is None, falls back to ``r = t`` (identical to teacher behavior up to
        the additive delta term, which is small at gate=0 and zero only when
        ``delta_embedder`` is deeply matched to ``timestep_embedder``).
        """
        r_timestep = self._r_timestep_pending
        if r_timestep is None:
            # Fall back: r = t, i.e. delta_t = 0 (for type 't-r') or delta_t = t
            # (for type 'r'). Either way this is the "no r information" case.
            r_timestep = timestep

        if self.deltatime_type == "r":
            delta_t = r_timestep
        else:  # 't-r'
            delta_t = timestep - r_timestep

        # Project both timesteps using the shared sinusoidal projector.
        timesteps_proj = self.time_proj(timestep)
        delta_proj = self.time_proj(delta_t)

        # Cast to the embedder's parameter dtype before passing through MLPs
        # (matches diffusers' upstream pattern).
        embedder_dtype = next(self.timestep_embedder.parameters()).dtype
        timesteps_proj = timesteps_proj.to(embedder_dtype)
        delta_proj = delta_proj.to(embedder_dtype)

        # Embed both. delta_embedder is independently trainable.
        temb = self.timestep_embedder(timesteps_proj)
        delta_emb = self.delta_embedder(delta_proj)

        # Gate-mix. gate is a buffer so we cast to the temb dtype safely.
        gate = self.delta_emb_gate.to(temb.dtype)
        rt_emb = (1.0 - gate) * temb + gate * delta_emb

        # Pooled-text embedding is added on top, same as the original module.
        pooled = self.text_embedder(pooled_projection)
        return rt_emb + pooled


# ---------------------------------------------------------------------------
# Public setup + context manager
# ---------------------------------------------------------------------------

def setup_flowmap_for_sd3(
    transformer: nn.Module,
    gate_value: float = 0.25,
    deltatime_type: str = "r",
) -> nn.Module:
    """Replace ``transformer.time_text_embed`` with the flow-map variant.

    Idempotent: calling this twice on the same transformer is a no-op (the
    second call detects the wrapper and returns immediately).

    Important: must be called BEFORE FSDP/DDP wrapping so the new
    ``delta_embedder`` parameters are picked up by the distributed wrapper.

    Args:
        transformer: A diffusers ``SD3Transformer2DModel`` (unwrapped).
        gate_value: Fixed gate mixing ratio (NOT learned). AnyFlow defaults
            to 0.25 across all released configs.
        deltatime_type: 'r' (default, embed r directly) or 't-r' (embed
            interval length).

    Returns:
        The same transformer, modified in-place.
    """
    if isinstance(transformer.time_text_embed, SD3FlowMapTimeTextEmbed):
        logger.info("setup_flowmap_for_sd3: already wrapped, skipping.")
        return transformer
    if not isinstance(transformer.time_text_embed, CombinedTimestepTextProjEmbeddings):
        raise TypeError(
            f"Expected transformer.time_text_embed to be CombinedTimestepTextProjEmbeddings, "
            f"got {type(transformer.time_text_embed).__name__}. SD3.5 flow-map wrapping "
            f"only supports the standard SD3Transformer2DModel."
        )
    original = transformer.time_text_embed
    transformer.time_text_embed = SD3FlowMapTimeTextEmbed(
        original=original,
        gate_value=gate_value,
        deltatime_type=deltatime_type,
    )
    logger.info(
        "Wrapped SD3.5 transformer.time_text_embed with flow-map two-time "
        "embedding (gate=%.3f, deltatime_type=%s).",
        gate_value, deltatime_type,
    )
    return transformer


def _unwrap_to_sd3_transformer(model: nn.Module) -> nn.Module:
    """Unwrap a possibly DDP/FSDP-wrapped model to the inner transformer."""
    # DDP wrapping
    inner = getattr(model, "module", model)
    # FSDP1 with use_orig_params=True keeps attribute access transparent;
    # we still need to reach the underlying nn.Module.
    return inner


@contextmanager
def with_r_timestep(model: nn.Module, r_timestep: Optional[torch.Tensor]):
    """Context manager that stashes ``r_timestep`` on the wrapped time embed.

    Trainers wrap each forward with this context so that the
    :class:`SD3FlowMapTimeTextEmbed` inside the transformer reads the right
    ``r`` for that particular call. Nesting is supported (the previous value
    is restored on exit).

    Safe with DDP/FSDP: we set the attribute on the unwrapped inner module
    (since DDP/FSDP forward delegates to the same underlying module).

    No-op if the model has not been wrapped via :func:`setup_flowmap_for_sd3`
    (for example, when calling the model in non-flow-map mode for ablation).

    Args:
        model: The transformer (DDP/FSDP-wrapped or not).
        r_timestep: Tensor of shape matching the timestep argument that will
            be passed to the transformer (e.g. ``[B]``). May be None.
    """
    inner = _unwrap_to_sd3_transformer(model)
    time_text_embed = getattr(inner, "time_text_embed", None)
    if not isinstance(time_text_embed, SD3FlowMapTimeTextEmbed):
        # Not wrapped: yield without doing anything. This makes the helper
        # safe to call before setup_flowmap_for_sd3 (e.g. in early init
        # where we run a sanity forward).
        yield
        return
    previous = time_text_embed._r_timestep_pending
    time_text_embed._r_timestep_pending = r_timestep
    try:
        yield
    finally:
        time_text_embed._r_timestep_pending = previous


# ---------------------------------------------------------------------------
# Predict-noise helper (mirrors predict_noise_sd35 with r_timestep injection)
# ---------------------------------------------------------------------------

def predict_noise_sd35_flowmap(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    timesteps: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    r_timesteps: torch.Tensor,
    guidance_scale: float = 1.0,
    uncond_text_embeddings: Optional[torch.Tensor] = None,
    uncond_pooled_prompt_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a SD3.5 flow-map transformer forward pass with optional CFG.

    Drop-in replacement for :func:`meanflownft.models.sd35.predict_noise_sd35`,
    with an extra ``r_timesteps`` argument injected via :func:`with_r_timestep`.
    When ``guidance_scale > 1.0``, performs CFG batched over the cond/uncond
    pair just like the base SD3.5 helper.

    Args:
        model: Flow-map-wrapped SD3 transformer (DDP/FSDP-wrapped OK).
        noisy_latents: ``[B, C, H, W]``.
        text_embeddings: Conditional text embeddings.
        timesteps: ``[B]`` raw model timesteps in [0, num_train_timesteps].
        pooled_prompt_embeds: Conditional pooled CLIP embeddings.
        r_timesteps: ``[B]`` flow-map "r" target timestep, must satisfy
            ``r <= t`` element-wise (not enforced here).
        guidance_scale: CFG scale; 1.0 means no CFG (single forward).
        uncond_text_embeddings / uncond_pooled_prompt_embeds: Required when
            ``guidance_scale > 1.0``.

    Returns:
        Predicted velocity ``[B, C, H, W]`` (CFG-applied if applicable).
    """
    use_cfg = guidance_scale > 1.0
    if use_cfg:
        if uncond_text_embeddings is None or uncond_pooled_prompt_embeds is None:
            raise ValueError(
                "predict_noise_sd35_flowmap: CFG requires uncond_text_embeddings "
                "and uncond_pooled_prompt_embeds."
            )
        model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        embeddings = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)
        ts = torch.cat([timesteps, timesteps], dim=0)
        pooled = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        r_ts = torch.cat([r_timesteps, r_timesteps], dim=0)
        with with_r_timestep(model, r_ts):
            pred = model(
                hidden_states=model_input,
                timestep=ts,
                encoder_hidden_states=embeddings,
                pooled_projections=pooled,
            ).sample
        pred_uncond, pred_cond = pred.chunk(2, dim=0)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    with with_r_timestep(model, r_timesteps):
        return model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_prompt_embeds,
        ).sample
