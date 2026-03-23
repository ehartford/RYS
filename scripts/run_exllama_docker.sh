#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

IMAGE_TAG=${IMAGE_TAG:-rys-exllama:cu128}
FLASH_ATTN_IMAGE=${FLASH_ATTN_IMAGE:-flashattn-cu128:py310}
EXLLAMAV3_PATH=${EXLLAMAV3_PATH:-}
MODEL_DIR=${MODEL_DIR:-}

QUEUE_FILE=${QUEUE_FILE:-results/scan/queue.json}
COMBINED_RESULTS_FILE=${COMBINED_RESULTS_FILE:-results/scan/combined_results.pkl}
MATH_RESULTS_FILE=${MATH_RESULTS_FILE:-results/scan/math_results.pkl}
EQ_RESULTS_FILE=${EQ_RESULTS_FILE:-results/scan/eq_results.pkl}

MATH_DATASET=${MATH_DATASET:-datasets/math_16.json}
EQ_DATASET=${EQ_DATASET:-datasets/eq_16.json}
MATH_MAX_NEW=${MATH_MAX_NEW:-64}
EQ_MAX_NEW=${EQ_MAX_NEW:-64}
CACHE_SIZE=${CACHE_SIZE:-0}
AUTO_CACHE=${AUTO_CACHE:-1}
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-2048}
MAX_OUTPUT_SIZE=${MAX_OUTPUT_SIZE:-0}
DEVICE=${DEVICE:-}
RESERVE_PER_DEVICE=${RESERVE_PER_DEVICE:-}
USE_PER_DEVICE=${USE_PER_DEVICE:-}
WORKER_ID=${WORKER_ID:-docker-exl3}
FORCE_BUILD_BASE=${FORCE_BUILD_BASE:-0}

if [ -z "${MODEL_DIR}" ]; then
  echo "MODEL_DIR is required." >&2
  exit 1
fi

if [ -z "${EXLLAMAV3_PATH}" ]; then
  echo "EXLLAMAV3_PATH is required." >&2
  exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if [ ! -d "${EXLLAMAV3_PATH}" ]; then
  echo "EXLLAMAV3_PATH does not exist: ${EXLLAMAV3_PATH}" >&2
  exit 1
fi

if [ "$FORCE_BUILD_BASE" = "1" ] || ! docker image inspect "$FLASH_ATTN_IMAGE" >/dev/null 2>&1; then
  MAX_JOBS=${MAX_JOBS:-8}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
  DOCKER_PLATFORM=${DOCKER_PLATFORM:-}
  MAX_JOBS="$MAX_JOBS" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" DOCKER_PLATFORM="$DOCKER_PLATFORM" \
    "${SCRIPT_DIR}/build_flashattn_image.sh"
fi

docker build -f "${REPO_ROOT}/docker/Dockerfile.exllama" \
  --build-arg BASE_IMAGE="$FLASH_ATTN_IMAGE" \
  -t "$IMAGE_TAG" \
  "${REPO_ROOT}"

RUN_FLAGS=(--rm -i)
if [ -t 1 ]; then
  RUN_FLAGS+=(-t)
fi

WORKER_CMD=(
  python /workspace/repo/scripts/run_exllama_math_eq_combined_worker.py
  --queue-file "/workspace/repo/${QUEUE_FILE}"
  --combined-results-file "/workspace/repo/${COMBINED_RESULTS_FILE}"
  --math-results-file "/workspace/repo/${MATH_RESULTS_FILE}"
  --eq-results-file "/workspace/repo/${EQ_RESULTS_FILE}"
  --model-dir "${MODEL_DIR}"
  --math-dataset-path "/workspace/repo/${MATH_DATASET}"
  --eq-dataset-path "/workspace/repo/${EQ_DATASET}"
  --math-max-new "${MATH_MAX_NEW}"
  --eq-max-new "${EQ_MAX_NEW}"
  --cache-size "${CACHE_SIZE}"
  --max-chunk-size "${MAX_CHUNK_SIZE}"
  --max-output-size "${MAX_OUTPUT_SIZE}"
  --worker-id "${WORKER_ID}"
)

if [ "${AUTO_CACHE}" = "1" ]; then
  WORKER_CMD+=(--auto-cache)
fi

if [ -n "${DEVICE}" ]; then
  WORKER_CMD+=(--device "${DEVICE}")
fi

if [ -n "${RESERVE_PER_DEVICE}" ]; then
  WORKER_CMD+=(--reserve-per-device "${RESERVE_PER_DEVICE}")
fi

if [ -n "${USE_PER_DEVICE}" ]; then
  WORKER_CMD+=(--use-per-device "${USE_PER_DEVICE}")
fi

docker run "${RUN_FLAGS[@]}" \
  --gpus all \
  --ipc=host \
  -e EXLLAMAV3_PATH=/workspace/exllamav3 \
  -e RYS_PATH=/workspace/repo \
  -e RYS_REPRO_PATH=/workspace/repo \
  -v "${REPO_ROOT}:/workspace/repo" \
  -v "${EXLLAMAV3_PATH}:/workspace/exllamav3" \
  -v "${MODEL_DIR}:${MODEL_DIR}:ro" \
  "$IMAGE_TAG" \
  "${WORKER_CMD[@]}"
