#!/bin/bash
# Generate the SD3.5-Medium MeanFlowNFT Stage 1/2 dataset (40 steps, CFG 4.5,
# 512x512) from the LAION aesthetic prompt list.
#
# Single-node usage:
#   bash scripts/generate_sd35_consistency_data.sh [NUM_GPUS] [PROMPT_FILE] [OUTPUT_DIR] [EXTRA_ARGS...]
#
# Multi-node usage: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT
# before launching the same command on every node.

set -euo pipefail

NUM_GPUS="${1:-8}"
PROMPT_FILE="${2:-dataset/laion_aes_6p5/train.txt}"
OUTPUT_DIR="${3:-${MEANFLOWNFT_DATA_DIR:-data/anyflow/sd35m_laion_aes_6p5_40step_cfg4.5_512}}"
EXTRA_ARGS=("${@:4}")
set --

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# NCCL tuning for multi-node generation.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

TOTAL_GPUS=$((NUM_GPUS * NNODES))
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

echo "=============================================="
echo "MeanFlowNFT Stage 1/2 Data Generation (SD3.5-Medium)"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  Total GPUs:  ${TOTAL_GPUS}"
echo "  Prompts:     ${PROMPT_FILE}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Recipe:      40 steps, CFG=4.5, 512x512"
echo "  Time:        $(date)"
echo "=============================================="

"${TORCHRUN_BIN}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/generate_consistency_data.py \
    --pretrained_path "${MEANFLOWNFT_SD35_PATH:-models/stable-diffusion-3.5-medium}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_inference_steps 40 \
    --guidance_scale 4.5 \
    --image_resolution 512 \
    --batch_size 16 \
    --dtype bf16 \
    "${EXTRA_ARGS[@]}"
