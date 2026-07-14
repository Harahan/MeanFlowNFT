#!/bin/bash
# MeanFlowNFT: SD3.5 MeanFlowNFT launch script.
#
# Loads the two-stage AnyFlow flow-map generator and applies MeanFlowNFT
# reinforcement learning in V-space (V_theta derived from u_theta via central
# difference). Stage 1 and Stage 2 LoRAs are merged in order before a fresh
# Stage 3 MeanFlowNFT LoRA is trained.
#
# Single-node usage:
#   bash scripts/train_sd35_meanflow_nft.sh [NUM_GPUS] [CONFIG_PATH] [EXTRA_ARGS...]

set -euo pipefail

NUM_GPUS="${1:-8}"
CONFIG="${2:-configs/meanflow_nft/sd35m_meanflow_nft.yaml}"
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
echo "MeanFlowNFT Training"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  Total GPUs:  ${TOTAL_GPUS}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG}"
echo "  Trainer:     meanflow_nft"
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
    --trainer meanflow_nft \
    "${EXTRA_ARGS[@]}"
