#!/usr/bin/env bash
# Compare a MeanFlowNFT stage prefix at several FlowMap rollout lengths.
#
# Usage:
#   NUM_STAGES=3 bash scripts/inference_meanflow_nft_steps.sh \
#     [CONFIG] [NUM_GPUS] [STEP ...]
#
# Defaults: the release SD3.5 config, one GPU, and steps 2 4 8 16 32.
# Set OUT_BASE, PYTHON_BIN, TORCHRUN_BIN, or MASTER_PORT in the environment
# when customization is needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CALLER_CWD="${PWD}"

DEFAULT_CONFIG="${REPO_ROOT}/configs/inference/sd35m_meanflow_nft.yaml"
CONFIG_INPUT="${1:-${DEFAULT_CONFIG}}"
NUM_PROCESSES="${2:-${NUM_GPUS:-1}}"
NUM_STAGES="${NUM_STAGES:-3}"
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
    echo "NUM_GPUS must be a positive integer, got: ${NUM_PROCESSES}" >&2
    exit 2
fi
if [[ ! "${NUM_STAGES}" =~ ^[1-3]$ ]]; then
    echo "NUM_STAGES must be 1, 2, or 3, got: ${NUM_STAGES}" >&2
    exit 2
fi
for step in "${STEPS[@]}"; do
    if [[ ! "${step}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Each step count must be a positive integer, got: ${step}" >&2
        exit 2
    fi
done

set --
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29555}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/inference_outputs/sd35m_meanflow_nft}"

echo "SD3.5-Medium MeanFlowNFT step comparison"
echo "  config: ${CONFIG}"
echo "  GPUs:   ${NUM_PROCESSES}"
echo "  stages: ${NUM_STAGES}"
echo "  steps:  ${STEPS[*]}"
echo "  output: ${OUT_BASE}"

cd -- "${REPO_ROOT}"
for step in "${STEPS[@]}"; do
    output_dir="${OUT_BASE}/steps_${step}"
    echo "Running ${step} steps -> ${output_dir}"
    if (( NUM_PROCESSES == 1 )); then
        "${PYTHON_BIN}" "${REPO_ROOT}/inference.py" "${CONFIG}" \
            --override "num_stages=${NUM_STAGES}" "num_steps=${step}" \
            "output_dir=${output_dir}"
    else
        "${TORCHRUN_BIN}" \
            --nproc_per_node="${NUM_PROCESSES}" \
            --master_port="${MASTER_PORT}" \
            "${REPO_ROOT}/inference.py" "${CONFIG}" \
            --override "distributed=true" "num_stages=${NUM_STAGES}" \
            "num_steps=${step}" \
            "output_dir=${output_dir}"
    fi
done

echo "Completed: ${OUT_BASE}"
