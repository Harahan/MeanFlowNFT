#!/bin/bash
# MeanFlowNFT Stage 2: SD3.5 AnyFlow on-policy distillation.
#
# Implements the on-policy distillation stage of the AnyFlow paper
# (arXiv 2605.13724) on SD3.5-Medium. Stage 1's generator LoRA is merged
# into the base before a fresh Stage 2 LoRA is trained; configure
# model.generator_lora.load_path in the YAML or with --override.
#
# Single-node usage:
#   bash scripts/train_sd35_anyflow_onpolicy.sh [NUM_GPUS] [CONFIG_PATH] [EXTRA_ARGS...]

set -euo pipefail

NUM_GPUS="${1:-8}"
CONFIG="${2:-configs/anyflow/sd35m_anyflow_onpolicy.yaml}"
EXTRA_ARGS=("${@:3}")
set --

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

TOTAL_GPUS=$((NUM_GPUS * NNODES))
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

echo "=============================================="
echo "MeanFlowNFT Stage 2 (AnyFlow On-Policy)"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  Total GPUs:  ${TOTAL_GPUS}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG}"
echo "  Trainer:     anyflow_onpolicy"
echo "  Time:        $(date)"
echo "=============================================="

"${TORCHRUN_BIN}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    "${CONFIG}" \
    --trainer anyflow_onpolicy \
    "${EXTRA_ARGS[@]}"
