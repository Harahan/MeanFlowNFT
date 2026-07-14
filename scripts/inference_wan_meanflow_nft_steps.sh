#!/usr/bin/env bash
# Generate the final Wan MeanFlowNFT policy at several FlowMap step counts.
#
# Usage:
#   MEANFLOWNFT_WAN_CKPT=/path/to/generator_ema.pt \
#   bash scripts/inference_wan_meanflow_nft_steps.sh \
#       [CONFIG] [NUM_GPUS] [STEP ...]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CALLER_CWD="${PWD}"

DEFAULT_CONFIG="${REPO_ROOT}/configs/inference/wan2.1_t2v_1.3b.yaml"
CONFIG_INPUT="${1:-${DEFAULT_CONFIG}}"
NUM_PROCESSES="${2:-${NUM_GPUS:-1}}"
if (( $# >= 3 )); then
    STEPS=("${@:3}")
else
    STEPS=(2 4 8 16 32)
fi

if [[ "${CONFIG_INPUT}" = /* ]]; then
    CONFIG="${CONFIG_INPUT}"
elif [[ -f "${CALLER_CWD}/${CONFIG_INPUT}" ]]; then
    CONFIG="${CALLER_CWD}/${CONFIG_INPUT}"
else
    CONFIG="${REPO_ROOT}/${CONFIG_INPUT}"
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Inference config not found: ${CONFIG}" >&2
    exit 2
fi
if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer: ${NUM_PROCESSES}" >&2
    exit 2
fi
if [[ -z "${MEANFLOWNFT_WAN_CKPT:-}" ]]; then
    echo "Set MEANFLOWNFT_WAN_CKPT to the final generator_ema.pt or adapter." >&2
    exit 2
fi
if [[ "${MEANFLOWNFT_WAN_CKPT}" = /* ]]; then
    CHECKPOINT="${MEANFLOWNFT_WAN_CKPT}"
else
    CHECKPOINT="${CALLER_CWD}/${MEANFLOWNFT_WAN_CKPT}"
fi
if [[ ! -e "${CHECKPOINT}" ]]; then
    echo "MeanFlowNFT checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi
for step in "${STEPS[@]}"; do
    if [[ ! "${step}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Each step count must be a positive integer: ${step}" >&2
        exit 2
    fi
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29555}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/inference_outputs/wan_meanflow_nft}"

echo "Wan2.1 MeanFlowNFT step comparison"
echo "  config:     ${CONFIG}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  GPUs:       ${NUM_PROCESSES}"
echo "  steps:      ${STEPS[*]}"
echo "  output:     ${OUT_BASE}"

cd -- "${REPO_ROOT}"
for step in "${STEPS[@]}"; do
    output_dir="${OUT_BASE}/steps_${step}"
    overrides=(
        "mode=meanflow_nft"
        "meanflow_nft_path=${CHECKPOINT}"
        "num_steps=${step}"
        "output_dir=${output_dir}"
    )
    if [[ -n "${MEANFLOWNFT_ANYFLOW_WAN_PATH:-}" ]]; then
        overrides+=(
            "anyflow_pretrained_path=${MEANFLOWNFT_ANYFLOW_WAN_PATH}"
        )
    fi
    echo "Running ${step} steps -> ${output_dir}/meanflow_nft"
    if (( NUM_PROCESSES == 1 )); then
        "${PYTHON_BIN}" "${REPO_ROOT}/inference.py" "${CONFIG}" \
            --override "${overrides[@]}"
    else
        "${TORCHRUN_BIN}" \
            --nproc_per_node="${NUM_PROCESSES}" \
            --master_port="${MASTER_PORT}" \
            "${REPO_ROOT}/inference.py" "${CONFIG}" \
            --override "${overrides[@]}"
    fi
done

echo "Completed: ${OUT_BASE}"
