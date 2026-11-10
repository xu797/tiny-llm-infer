#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${repo_root}/build/model/model_main"
tokenizer="${repo_root}/Qwen2.5-3B-Instruct/tokenizer.json"
model="${repo_root}/Qwen2.5-3B-Instruct"

# cmake -S "${repo_root}" -B "${repo_root}/build" -DCMAKE_BUILD_TYPE=Release
# cmake --build "${repo_root}/build" -j"${BUILD_JOBS:-2}"

# if command -v nvidia-smi >/dev/null 2>&1; then
#     nvidia-smi
# else
#     echo "nvidia-smi not found; continuing and letting CUDA initialization report device errors." >&2
# fi

exec "${binary}" --bench \
    --tokenizer "${tokenizer}" \
    --model "${model}" \
    --device cuda \
    --num-seqs "${NUM_SEQS:-8}" \
    --min-input-len "${MIN_INPUT_LEN:-64}" \
    --max-input-len "${MAX_INPUT_LEN:-128}" \
    --min-output-len "${MIN_OUTPUT_LEN:-32}" \
    --max-output-len "${MAX_OUTPUT_LEN:-64}" \
    --warmup "${WARMUP:-1}" \
    --repeats "${REPEATS:-3}" \
    --max-seq-len "${MAX_SEQ_LEN:-256}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.7}" \
    --quantization "${QUANTIZATION:-none}" \
    --manifest-prefix "${MANIFEST_PREFIX:-}" \
    --seed "${SEED:-0}" \
    "$@"
