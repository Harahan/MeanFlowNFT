#!/usr/bin/env bash
# Launch Wan2.1 MeanFlowNFT on four nodes.
#
# Set NODE_RANK=0..3 and MASTER_ADDR to the rank-0 host on every node.

set -euo pipefail

NUM_GPUS="${1:-8}"
CONFIG="${2:-configs/meanflow_nft/wan2.1_t2v_1.3b_meanflow_nft.yaml}"
EXTRA_ARGS=("${@:3}")
set --

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${NNODES}" != "4" ]]; then
    echo "This release launcher requires NNODES=4; got ${NNODES}." >&2
    exit 2
fi
if [[ -z "${NODE_RANK}" || -z "${MASTER_ADDR}" ]]; then
    echo "Set NODE_RANK (0..3) and MASTER_ADDR before launching." >&2
    exit 2
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export MEANFLOWNFT_DIST_TIMEOUT_HOURS="${MEANFLOWNFT_DIST_TIMEOUT_HOURS:-6}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

echo "============================================================"
echo "Wan2.1 MeanFlowNFT"
echo "  nodes:       ${NNODES} (rank ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  config:      ${CONFIG}"
echo "============================================================"

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
