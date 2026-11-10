#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${repo_root}/benchmark/run_cuda_benchmark.sh"
profiles=(
    "batch1_prompt128,1,128,128"
    "batch8_prompt128,8,128,128"
    "batch32_prompt128,32,128,128"
    # "batch1_prompt2048,1,2048,2048"
    # "batch8_prompt1024,8,1024,1024"
)

printf 'profile,quantization,batch_size,prompt_tokens,median_output_tokens_per_second\n'
if [[ -n "${MANIFEST_DIR:-}" ]]; then
    mkdir -p "${MANIFEST_DIR}"
fi
for profile in "${profiles[@]}"; do
    IFS=, read -r name batch min_prompt max_prompt <<< "${profile}"
    if [[ "${min_prompt}" == "${max_prompt}" ]]; then
        prompt_label="${min_prompt}"
    else
        prompt_label="${min_prompt}-${max_prompt}"
    fi
    manifest_prefix=""
    if [[ -n "${MANIFEST_DIR:-}" ]]; then
        manifest_prefix="${MANIFEST_DIR}/${name}"
    fi

    for quantization in none; do
        if ! output="$(
            NUM_SEQS="${batch}" \
            MIN_INPUT_LEN="${min_prompt}" \
            MAX_INPUT_LEN="${max_prompt}" \
            MIN_OUTPUT_LEN=128 \
            MAX_OUTPUT_LEN=128 \
            WARMUP=3 \
            REPEATS=5 \
            MAX_SEQ_LEN=2176 \
            QUANTIZATION="${quantization}" \
            MANIFEST_PREFIX="${manifest_prefix}" \
            "${runner}"
        )"; then
            printf '%s\n' "${output}" >&2
            exit 1
        fi

        rates="$(printf '%s\n' "${output}" | awk '
            /^run=/ {
                for (i = 1; i <= NF; ++i) {
                    if ($i ~ /^output_tokens_per_second=/) {
                        split($i, value, "="); print value[2]
                    }
                }
            }
        ' | sort -n)"
        median="$(printf '%s\n' "${rates}" | sed -n '3p')"
        if [[ -z "${median}" ]]; then
            printf 'Could not read five measured throughput values for %s (%s).\n' \
                "${name}" "${quantization}" >&2
            exit 1
        fi
        printf '%s,%s,%s,%s,%s\n' \
            "${name}" "${quantization}" "${batch}" "${prompt_label}" "${median}"
    done
done
