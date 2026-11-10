#!/usr/bin/env python3
"""Run vLLM on TinyLLM benchmark request manifests and emit comparable CSV."""

import argparse
import csv
from contextlib import redirect_stdout
import glob
import json
import os
import re
import statistics
import sys
import time


def load_profiles(manifest_dir, repeats):
    grouped = {}
    for path in glob.glob(os.path.join(manifest_dir, "*_run*.json")):
        match = re.match(r"^(.*)_run(\d+)\.json$", os.path.basename(path))
        if not match:
            continue
        profile, run_number = match.group(1), int(match.group(2))
        with open(path, "r", encoding="utf-8") as manifest_file:
            requests = json.load(manifest_file)["requests"]
        grouped.setdefault(profile, {})[run_number] = requests

    if not grouped:
        raise ValueError(f"No *_run*.json manifests found in {manifest_dir}")
    for profile, runs in grouped.items():
        missing = sorted(set(range(1, repeats + 1)) - set(runs))
        if missing:
            raise ValueError(f"{profile} is missing run manifests: {missing}")
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model directory")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable vLLM CUDA graph capture and replay")
    args = parser.parse_args()

    profiles = load_profiles(args.manifest_dir, args.repeats)

    # Avoid a FlashInfer JIT build that may select an older system nvcc than
    # the CUDA version used to build the installed PyTorch/vLLM wheels.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise SystemExit("vLLM is not installed in this Python environment.") from error

    with redirect_stdout(sys.stderr):
        llm = LLM(
            model=args.model,
            dtype="half",
            quantization=None,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=args.enforce_eager,
        )

    def run_requests(requests):
        prompts = [
            {"prompt_token_ids": request["prompt_token_ids"]}
            for request in requests
        ]
        sampling = [
            SamplingParams(
                max_tokens=request["max_tokens"],
                temperature=request["temperature"],
                top_p=request["top_p"],
                ignore_eos=request["ignore_eos"],
                seed=request["seed"],
            )
            for request in requests
        ]
        with redirect_stdout(sys.stderr):
            return llm.generate(prompts, sampling, use_tqdm=False)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("engine", "profile", "run", "requests", "output_tokens",
                     "elapsed_seconds", "output_tokens_per_second"))
    for profile in sorted(profiles):
        runs = profiles[profile]
        for _ in range(args.warmup):
            run_requests(runs[1])

        rates = []
        for run_number in range(1, args.repeats + 1):
            requests = runs[run_number]
            started = time.perf_counter()
            outputs = run_requests(requests)
            elapsed = time.perf_counter() - started
            output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            rate = output_tokens / elapsed
            rates.append(rate)
            writer.writerow(("vllm", profile, run_number, len(requests), output_tokens,
                             f"{elapsed:.6f}", f"{rate:.4f}"))
            sys.stdout.flush()

        writer.writerow(("vllm", profile, "median", len(runs[1]), "",
                         "", f"{statistics.median(rates):.4f}"))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
