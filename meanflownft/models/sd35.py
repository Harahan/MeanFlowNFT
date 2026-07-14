"""
SD3.5 Model Loading Factory.

Handles loading all components of the Stable Diffusion 3.5 pipeline:
- SD3Transformer2DModel (the DiT backbone)
- AutoencoderKL (VAE for latent encoding/decoding)
- Text encoders: 2x CLIP + 1x T5
- Tokenizers
- FlowMatchEulerDiscreteScheduler

Reference: SenseFlow trainer_sd35_senseflow.py for SD3.5-specific loading.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from meanflownft.parallel.utils import is_main_process
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    PretrainedConfig,
)

import torch.nn as nn

from meanflownft.config import ModelConfig
from meanflownft.utils.fast_init import fast_init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference utility functions (shared by trainers and inference)
# ---------------------------------------------------------------------------

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract values from 1D tensor `a` at indices `t`, reshaped for broadcasting.

    Args:
        a: 1D source tensor (e.g., sigma schedule).
        t: 1D index tensor of shape [B].
        x_shape: Target shape for broadcasting (e.g., [B, C, H, W]).

    Returns:
        Tensor of shape [B, 1, 1, ...] matching len(x_shape) dimensions.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predict_noise_sd35(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    timesteps: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance_scale: float = 1.0,
    uncond_text_embeddings: torch.Tensor | None = None,
    uncond_pooled_prompt_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a forward pass through the SD3.5 transformer with optional CFG.

    When guidance_scale > 1.0, performs classifier-free guidance by running
    conditional and unconditional forward passes and interpolating.

    Args:
        model: SD3Transformer2DModel instance.
        noisy_latents: Noisy input latents [B, C, H, W].
        text_embeddings: Conditional text embeddings [B, seq_len, dim].
        timesteps: Diffusion timesteps [B].
        pooled_prompt_embeds: Pooled CLIP embeddings [B, pooled_dim].
        guidance_scale: CFG scale. 1.0 = no guidance.
        uncond_text_embeddings: Unconditional text embeddings for CFG.
        uncond_pooled_prompt_embeds: Unconditional pooled embeddings for CFG.

    Returns:
        Predicted noise/velocity [B, C, H, W].
    """
    use_cfg = guidance_scale > 1.0

    if use_cfg:
        assert uncond_text_embeddings is not None and uncond_pooled_prompt_embeds is not None, \
            "CFG requires unconditional embeddings"
        model_input = torch.cat([noisy_latents, noisy_latents])
        embeddings = torch.cat([uncond_text_embeddings, text_embeddings])
        ts = torch.cat([timesteps, timesteps])
        pooled = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds])

        pred = model(
            model_input, timestep=ts,
            encoder_hidden_states=embeddings,
            pooled_projections=pooled,
        ).sample
        pred_uncond, pred_cond = pred.chunk(2)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    else:
        return model(
            noisy_latents, timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_prompt_embeds,
        ).sample


def recover_noise_sd35(
    x_sigma: torch.Tensor,
    v_pred: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Recover noise from velocity prediction.

    For flow matching: x_sigma = (1-sigma)*x0 + sigma*eps
    Velocity: v = eps - x0
    Therefore: eps = x_sigma + (1-sigma)*v

    Args:
        x_sigma: Noisy sample [B, C, H, W].
        v_pred: Velocity prediction [B, C, H, W].
        sigma: Noise level (scalar or broadcastable tensor).

    Returns:
        Recovered noise [B, C, H, W].
    """
    return (x_sigma.float() + (1 - sigma) * v_pred.float()).to(x_sigma.dtype)


def compute_sigmas_sd35(
    denoising_step_list: list[int],
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
) -> list[float]:
    """Compute shifted sigmas from denoising step list for SD3.5.

    Applies the same time-shift formula as FlowMatchEulerDiscreteScheduler:
        raw_sigma = step / num_train_timesteps
        shifted_sigma = shift * raw_sigma / (1 + (shift - 1) * raw_sigma)

    Args:
        denoising_step_list: Raw denoising steps, e.g. [1000, 750, 500, 250].
        num_train_timesteps: Total training timesteps (default 1000).
        shift: Time-shift parameter (SD3.5 medium default 3.0).

    Returns:
        List of shifted sigmas, e.g. [1.0, 0.9, 0.75, 0.5].
    """
    sigmas = []
    for step in denoising_step_list:
        raw_sigma = step / num_train_timesteps
        shifted = shift * raw_sigma / (1 + (shift - 1) * raw_sigma)
        sigmas.append(round(shifted, 6))
    return sigmas


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype_str}. Choose from: {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# Text encoder class resolver
# ---------------------------------------------------------------------------

def _import_text_encoder_cls(pretrained_path: str, subfolder: str) -> type:
    """Dynamically resolve the text encoder class from pretrained config.

    Args:
        pretrained_path: Path to the pretrained model directory.
        subfolder: Subfolder name (e.g., "text_encoder", "text_encoder_2", "text_encoder_3").

    Returns:
        The appropriate text encoder class.
    """
    config = PretrainedConfig.from_pretrained(pretrained_path, subfolder=subfolder)
    arch = config.architectures[0]
    if arch == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    elif arch == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported text encoder architecture: {arch}")


# ---------------------------------------------------------------------------
# Text encoding functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_clip(
    text_encoder: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    prompts: list[str],
    device: torch.device,
    max_length: int = 77,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts using a CLIP text encoder.

    Args:
        text_encoder: CLIP text encoder model.
        tokenizer: Corresponding CLIP tokenizer.
        prompts: List of text prompts.
        device: Target device.
        max_length: Max token sequence length.

    Returns:
        Tuple of (hidden_states, pooled_output):
        - hidden_states: [B, seq_len, dim] from second-to-last layer
        - pooled_output: [B, dim] pooled CLS embedding
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    outputs = text_encoder(
        text_inputs.input_ids.to(device),
        output_hidden_states=True,
    )
    # Use second-to-last hidden state (standard for CLIP conditioning)
    hidden_states = outputs.hidden_states[-2].to(dtype=text_encoder.dtype, device=device)
    pooled_output = outputs[0]
    return hidden_states, pooled_output


@torch.no_grad()
def encode_prompt_t5(
    text_encoder: T5EncoderModel,
    tokenizer: T5TokenizerFast,
    prompts: list[str],
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """Encode prompts using a T5 text encoder.

    Args:
        text_encoder: T5 encoder model.
        tokenizer: Corresponding T5 tokenizer.
        prompts: List of text prompts.
        device: Target device.
        max_length: Max token sequence length.

    Returns:
        Encoder hidden states: [B, seq_len, dim].
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    outputs = text_encoder(text_inputs.input_ids.to(device))
    return outputs[0].to(dtype=text_encoder.dtype, device=device)


@torch.no_grad()
def encode_prompts_sd35(
    prompts: list[str],
    text_encoders: list,
    tokenizers: list,
    device: torch.device,
    max_sequence_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts using the full SD3.5 text encoder stack (2xCLIP + T5).

    Follows the SD3.5 conditioning format: CLIP hidden states are concatenated
    along the feature dim, then padded and concatenated with T5 along the
    sequence dim. Pooled CLIP outputs are concatenated for the pooled projection.

    Args:
        prompts: List of text prompts.
        text_encoders: [clip_encoder_1, clip_encoder_2, t5_encoder].
        tokenizers: [clip_tokenizer_1, clip_tokenizer_2, t5_tokenizer].
        device: Target device.
        max_sequence_length: T5 token sequence length (default 128 matches the
            MeanFlowNFT historical default; SD3 pipeline uses 256). The output
            ``prompt_embeds`` will have shape ``[B, 77 + max_sequence_length, dim]``
            (CLIP is always padded to 77 by the model architecture). Callers
            consuming pre-generated SD3 pipeline data should pass ``256`` so
            the on-the-fly uncond / eval encoding matches the dataset's
            ``prompt_embeds`` sequence length.

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds):
        - prompt_embeds: [B, 77 + max_sequence_length, dim] combined text embeddings
        - pooled_prompt_embeds: [B, pooled_dim] pooled embeddings
    """
    # Encode with both CLIP models
    clip_embeds_list = []
    pooled_list = []
    for encoder, tokenizer in zip(text_encoders[:2], tokenizers[:2]):
        hidden, pooled = encode_prompt_clip(encoder, tokenizer, prompts, device)
        clip_embeds_list.append(hidden)
        pooled_list.append(pooled)

    # Concatenate CLIP hidden states along feature dimension
    clip_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_embeds = torch.cat(pooled_list, dim=-1)

    # Encode with T5
    t5_embeds = encode_prompt_t5(
        text_encoders[2], tokenizers[2], prompts, device,
        max_length=max_sequence_length,
    )

    # Pad CLIP embeddings to match T5 feature dimension, then concatenate along sequence dim
    clip_embeds = torch.nn.functional.pad(
        clip_embeds,
        (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

    return prompt_embeds, pooled_embeds


# ---------------------------------------------------------------------------
# Main model loading function
# ---------------------------------------------------------------------------

def load_sd35_models(config: ModelConfig) -> dict[str, Any]:
    """Load all SD3.5 pipeline components from a pretrained checkpoint.

    Loads the transformer, VAE, text encoders, tokenizers, and scheduler.
    Models are loaded to CPU first; the trainer handles device placement
    and FSDP/DDP wrapping.

    Args:
        config: Model configuration with pretrained_path and dtype.

    Returns:
        Dict with keys:
        - "transformer": SD3Transformer2DModel
        - "vae": AutoencoderKL
        - "text_encoders": [clip_1, clip_2, t5]
        - "tokenizers": [clip_tok_1, clip_tok_2, t5_tok]
        - "scheduler": FlowMatchEulerDiscreteScheduler
    """
    path = config.pretrained_path
    dtype = _get_torch_dtype(config.dtype)
    if is_main_process():
        logger.info(f"Loading SD3.5 models from: {path} (dtype={config.dtype})")

    # Check flash attention availability
    # Use torch's built-in check which is more reliable than importing flash_attn
    # directly (flash_attn import can fail under torchrun due to CUDA init race).
    attn_implementation = "eager"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
            if is_main_process():
                logger.info("  Flash Attention 2 available, enabling for all transformer models")
        except ImportError:
            attn_implementation = "sdpa"
            if is_main_process():
                logger.info("  Using PyTorch SDPA attention (flash_attn package not found)")
    else:
        if is_main_process():
            logger.warning("  Falling back to eager attention (PyTorch too old for SDPA)")

    # Use fast_init to skip redundant weight initialization during from_pretrained().
    # This avoids RNG consumption and speeds up loading (~2x for large models).
    with fast_init(torch.device("cpu")):
        # Transformer (the DiT backbone)
        transformer = SD3Transformer2DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        if is_main_process():
            logger.info(
                f"  Transformer loaded: {sum(p.numel() for p in transformer.parameters()) / 1e6:.1f}M params"
            )

        # VAE
        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
        if is_main_process():
            logger.info("  VAE loaded")

        # Text encoders
        text_encoder_1 = _import_text_encoder_cls(path, "text_encoder").from_pretrained(
            path, subfolder="text_encoder", torch_dtype=dtype,
        )
        text_encoder_2 = _import_text_encoder_cls(path, "text_encoder_2").from_pretrained(
            path, subfolder="text_encoder_2", torch_dtype=dtype,
        )
        text_encoder_3 = _import_text_encoder_cls(path, "text_encoder_3").from_pretrained(
            path, subfolder="text_encoder_3", torch_dtype=dtype,
        )
        if is_main_process():
            logger.info("  Text encoders loaded (2x CLIP + T5)")

    # Tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2")
    tokenizer_3 = T5TokenizerFast.from_pretrained(path, subfolder="tokenizer_3")
    if is_main_process():
        logger.info("  Tokenizers loaded")

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
    if is_main_process():
        logger.info("  Scheduler loaded (FlowMatchEulerDiscrete)")

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoders": [text_encoder_1, text_encoder_2, text_encoder_3],
        "tokenizers": [tokenizer_1, tokenizer_2, tokenizer_3],
        "scheduler": scheduler,
    }
