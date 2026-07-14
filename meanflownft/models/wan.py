"""
Wan2.1-T2V (AnyFlow flow-map) Model Loading Factory.

Video counterpart of :mod:`meanflownft.models.sd35`. Loads all components of the
AnyFlow-Wan2.1-T2V-1.3B-Diffusers pipeline:

- WanAnyFlowTransformer3DModel (vendored bidirectional flow-map DiT, two-time
  ``(t, r)`` conditioning baked in; see :mod:`meanflownft.models.wan_transformer`)
- AutoencoderKLWan (3D causal video VAE, spatial /8, temporal /4, z_dim=16)
- UMT5EncoderModel + T5TokenizerFast (text encoder; NO pooled embeddings)
- FlowMapScheduler (shift=5, == AnyFlow FlowMapDiscreteScheduler)

Mirrors the ``load_<model_type>_models(config) -> dict`` interface so the NFT
trainer family can dispatch on ``model_type`` without further changes.

Key differences from the SD3.5 / FLUX2 image backends:

- Latents are 5D ``[B, F, C, H, W]`` (frames-before-channels), e.g.
  ``[B, 21, 16, 60, 104]`` for 81 frames at 480x832 (temporal /4, spatial /8).
- The transformer's ``forward`` natively accepts a per-frame ``r_timestep``
  (the flow-map second time), so no ``with_r_timestep`` wrapper is needed
  (unlike SD3.5's ``time_text_embed`` monkey-wrap).
- Text conditioning is a single UMT5 sequence embedding ``[B, L, 4096]`` with
  NO pooled projection; callers must thread ``pooled_embeds=None`` through.
- The VAE is loaded in fp32 (decode fidelity; matches the GenRL reference) and
  uses per-channel ``latents_mean`` / ``latents_std`` (de)normalization.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel

from meanflownft.config import ModelConfig
from meanflownft.models.wan_transformer import WanAnyFlowTransformer3DModel
from meanflownft.parallel.utils import is_main_process
from meanflownft.schedulers.flowmap_scheduler import FlowMapScheduler
from meanflownft.utils.fast_init import fast_init

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. Choose from: {list(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# Inference utilities (shared by trainer + inference), video / flow-map flavored
# ---------------------------------------------------------------------------

def _broadcast_per_frame(
    timesteps: torch.Tensor, batch_size: int, num_frames: int, device, dtype,
) -> torch.Tensor:
    """Broadcast a scalar / ``[B]`` timestep to the per-frame ``[B, F]`` form the
    Wan transformer's ``condition_embedder`` expects.

    Accepts:
      - 0-d scalar           -> expanded to ``[B, F]``
      - ``[B]`` per-sample    -> repeated to ``[B, F]``
      - ``[B, F]`` per-frame  -> passed through (validated)
    """
    t = timesteps.to(device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.expand(batch_size).unsqueeze(-1).repeat(1, num_frames)
    elif t.ndim == 1:
        t = t.unsqueeze(-1).repeat(1, num_frames)
    elif t.ndim == 2:
        if t.shape != (batch_size, num_frames):
            raise ValueError(
                f"per-frame timestep shape {tuple(t.shape)} != "
                f"(batch={batch_size}, frames={num_frames})"
            )
    else:
        raise ValueError(f"unexpected timestep ndim={t.ndim}")
    return t


def predict_noise_wan(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    timesteps: torch.Tensor,
    r_timesteps: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    uncond_text_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass through the Wan AnyFlow flow-map transformer (bidirectional).

    Args:
        model: ``WanAnyFlowTransformer3DModel`` (DDP/FSDP-wrapped OK).
        noisy_latents: ``[B, F, C, H, W]`` latents (frames-before-channels).
        text_embeddings: UMT5 sequence embeds ``[B, L, 4096]``.
        timesteps: raw model timesteps, scalar / ``[B]`` / ``[B, F]``.
        r_timesteps: flow-map second time; defaults to ``timesteps`` (== single
            instantaneous velocity at ``r = t``).
        guidance_scale: CFG scale. AnyFlow-Wan bakes CFG in (inference is
            CFG-free, guidance_scale=1.0); >1.0 runs the dual-forward path.
        uncond_text_embeddings: required when ``guidance_scale > 1.0``.

    Returns:
        Predicted velocity ``[B, F, C, H, W]``.
    """
    if r_timesteps is None:
        r_timesteps = timesteps

    batch_size, num_frames = noisy_latents.shape[0], noisy_latents.shape[1]
    device, dtype = noisy_latents.device, torch.float32
    t_pf = _broadcast_per_frame(timesteps, batch_size, num_frames, device, dtype)
    r_pf = _broadcast_per_frame(r_timesteps, batch_size, num_frames, device, dtype)

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        if uncond_text_embeddings is None:
            raise ValueError("predict_noise_wan: CFG requires uncond_text_embeddings.")
        model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        embeddings = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)
        t_in = torch.cat([t_pf, t_pf], dim=0)
        r_in = torch.cat([r_pf, r_pf], dim=0)
        pred = model(
            hidden_states=model_input,
            timestep=t_in,
            r_timestep=r_in,
            encoder_hidden_states=embeddings,
            is_causal=False,
        ).sample
        pred_uncond, pred_cond = pred.chunk(2, dim=0)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    return model(
        hidden_states=noisy_latents,
        timestep=t_pf,
        r_timestep=r_pf,
        encoder_hidden_states=text_embeddings,
        is_causal=False,
    ).sample


# ---------------------------------------------------------------------------
# Text encoding (UMT5, no pooled) — mirrors the Wan pipeline exactly
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompts_wan(
    prompts: list[str],
    text_encoder: UMT5EncoderModel,
    tokenizer: Any,
    device: torch.device,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, None]:
    """Encode prompts with the Wan UMT5 text encoder.

    Replicates ``WanPipeline._get_t5_prompt_embeds``: tokenize to a fixed
    ``max_sequence_length``, run UMT5, then mask out padding by trimming each
    sequence to its true length and zero-padding back. Returns ``(embeds, None)``
    (Wan has no pooled projection).

    Returns:
        prompt_embeds: ``[B, max_sequence_length, 4096]``.
        None: pooled placeholder (Wan has no pooled embeddings).
    """
    prompts = [str(p) for p in prompts]
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    out = text_encoder(input_ids, attention_mask=mask).last_hidden_state
    out = out.to(dtype=text_encoder.dtype, device=device)
    # Trim to true length then zero-pad to max (Wan convention).
    trimmed = [u[:v] for u, v in zip(out, seq_lens)]
    prompt_embeds = torch.stack(
        [F.pad(u, (0, 0, 0, max_sequence_length - u.size(0))) for u in trimmed], dim=0
    )
    return prompt_embeds, None


# ---------------------------------------------------------------------------
# Video VAE encode / decode (AutoencoderKLWan)
# ---------------------------------------------------------------------------

def _wan_latents_mean_std(vae) -> tuple[torch.Tensor, torch.Tensor]:
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1)
    return mean, std


@torch.no_grad()
def decode_wan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode Wan latents ``[B, F, C, H, W]`` to pixel video ``[B, F, 3, Hp, Wp]``
    in ``[0, 1]``.

    De-normalizes with the VAE's per-channel ``latents_mean`` / ``latents_std``
    (diffusers WanPipeline convention) and decodes one sample at a time to keep
    peak memory bounded (3D video VAE decode is heavy).
    """
    device = latents.device
    if next(vae.parameters()).device != device:
        vae.to(device)
    vae_dtype = next(vae.parameters()).dtype

    # [B, F, C, H, W] -> [B, C, F, H, W]
    latents = rearrange(latents, "b f c h w -> b c f h w").to(vae_dtype)
    mean, std = _wan_latents_mean_std(vae)
    mean = mean.to(device=device, dtype=vae_dtype)
    std = std.to(device=device, dtype=vae_dtype)
    latents = latents / std + mean

    frames = []
    for i in range(latents.shape[0]):
        sample = vae.decode(latents[i : i + 1], return_dict=False)[0]  # [1, 3, Fp, Hp, Wp]
        frames.append(sample)
    video = torch.cat(frames, dim=0)  # [B, 3, Fp, Hp, Wp]
    video = (video / 2 + 0.5).clamp(0, 1)
    # [B, 3, Fp, Hp, Wp] -> [B, Fp, 3, Hp, Wp]
    video = rearrange(video, "b c f h w -> b f c h w")
    return video.float()


# ---------------------------------------------------------------------------
# Main model loading
# ---------------------------------------------------------------------------

def load_wan_models(config: ModelConfig) -> dict[str, Any]:
    """Load all AnyFlow-Wan2.1-T2V components from a diffusers checkpoint.

    Returns a dict with the same keys as :func:`meanflownft.models.sd35.load_sd35_models`:
        - "transformer": WanAnyFlowTransformer3DModel
        - "vae": AutoencoderKLWan (fp32)
        - "text_encoders": [UMT5EncoderModel]
        - "tokenizers": [T5TokenizerFast]
        - "scheduler": FlowMapScheduler(shift=5)
    """
    path = config.pretrained_path
    dtype = _get_torch_dtype(config.dtype)
    if is_main_process():
        logger.info(f"Loading Wan AnyFlow models from: {path} (dtype={config.dtype})")

    with fast_init(torch.device("cpu")):
        transformer = WanAnyFlowTransformer3DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=dtype,
        )
        if is_main_process():
            n_params = sum(p.numel() for p in transformer.parameters()) / 1e6
            logger.info(f"  Transformer loaded: {n_params:.1f}M params")
            logger.info(
                "  flow-map: deltatime_type=%s, gate_value=%s",
                getattr(transformer.config, "deltatime_type", None),
                getattr(transformer.config, "gate_value", None),
            )

        # VAE in fp32 for decode fidelity (matches GenRL reference).
        vae = AutoencoderKLWan.from_pretrained(
            path, subfolder="vae", torch_dtype=torch.float32,
        )
        if is_main_process():
            logger.info("  VAE loaded (AutoencoderKLWan, fp32)")

        text_encoder = UMT5EncoderModel.from_pretrained(
            path, subfolder="text_encoder", torch_dtype=dtype,
        )
        if is_main_process():
            logger.info("  Text encoder loaded (UMT5)")

    tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
    if is_main_process():
        logger.info("  Tokenizer loaded")

    # FlowMapScheduler == AnyFlow FlowMapDiscreteScheduler (shift=5).
    scheduler = FlowMapScheduler(
        num_train_timesteps=1000, shift=5.0, weight_type="uniform",
    )
    if is_main_process():
        logger.info("  Scheduler ready (FlowMapScheduler, shift=5)")

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoders": [text_encoder],
        "tokenizers": [tokenizer],
        "scheduler": scheduler,
    }
