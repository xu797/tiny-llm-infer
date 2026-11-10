# tiny-llm-infer

A small C++ inference engine with Qwen2.5/Qwen3 prompt batching, FP16 inference, and a preallocated paged KV cache.

## Build

Run from the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j2

Run the C++ and CUDA tests with:

    ctest --test-dir build --output-on-failure

<!-- ## Model Files

The examples use `Qwen2.5-3B-Instruct`. The model directory must contain the Hugging Face configuration and safetensors checkpoint files; pass the tokenizer JSON separately with `--tokenizer`. Single-file and indexed/sharded safetensors checkpoints are supported.

The runtime stores weights, activations, and KV cache in FP16. BF16 checkpoint weights are converted to FP16 during loading. CUDA projection layers use cuBLAS FP16 GEMM; attention and normalization reductions accumulate in FP32. -->

## Qwen2.5 inference

Single request:

    ./build/model/model_main --infer \
      --tokenizer Qwen2.5-3B-Instruct/tokenizer.json \
      --model Qwen2.5-3B-Instruct \
      --prompt $'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n' \
      --device cuda --max-seq-len 2048 --max-new-tokens 64

Pass --prompt more than once to submit a batch:

    ./build/model/model_main --infer \
      --tokenizer Qwen2.5-3B-Instruct/tokenizer.json \
      --model Qwen2.5-3B-Instruct \
      --prompt 'hello' --prompt 'introduce qwen2.5 model.' \
      --device cuda --max-seq-len 2048 --max-new-tokens 64

The CLI treats each prompt as text to continue; it does not apply tokenizer_config.json's chat template automatically. For instruction-style replies, include the model's chat markers in the prompt. Use `--device cpu` for CPU inference. Active sequences are decoded together. Prompt prefill processes multiple tokens per step and can be chunked to fit the scheduler token budget.

<!-- ## Continuous batching

`LLMEngine` supports online request submission: call `add_request()` while the engine is serving, then repeatedly call `step()`. The scheduler places active decode tokens first and uses the remaining token budget for newly arrived prompt prefill, so both phases can run in one model batch. Completed requests are returned by `step()`. Submit requests from another thread if arrivals happen during a model step; keep one thread responsible for calling `step()`. -->

## Benchmarking

`benchmark/run_cuda_benchmark.sh` runs one configurable CUDA benchmark. The matrix script runs the configured profiles with 3 warmups and 5 measured repeats, then prints one CSV row per profile using the median end-to-end output-token throughput:

    ./benchmark/run_quantization_matrix.sh > tiny.csv

The current matrix contains batch sizes 1, 8, and 32 with 128-token prompts and 128-token outputs. Requests use random categorical sampling (`temperature=0.6`, `top_p=1.0`) and ignore EOS. `output_tokens_per_second` is the total number of output tokens across the batch divided by elapsed time, not per-request speed. Each repeat gets different deterministic prompt tokens to avoid measuring prefix-cache reuse.

Set `MANIFEST_DIR` to save each measured repeat's exact prompt token IDs and sampling parameters. These manifests let the vLLM script run the same requests:

    MANIFEST_DIR=/tmp/tiny-vllm-manifests \
      ./benchmark/run_benchmark.sh > tiny.csv

### vLLM Comparison

Create a Python environment with vLLM installed:

    uv venv vllm-env
    source vllm-env/bin/activate
    uv pip install vllm

Run vLLM against the manifests using FP16 weights, the same scheduler token budget, disabled prefix caching, and 3 warmups / 5 measured repeats:

    vllm-env/bin/python benchmark/run_vllm_comparison.py \
      --model Qwen2.5-3B-Instruct \
      --manifest-dir /tmp/tiny-vllm-manifests \
      > vllm.csv

Use `--enforce-eager` for an additional eager-mode measurement:

    vllm-env/bin/python benchmark/run_vllm_comparison.py \
      --model Qwen2.5-3B-Instruct \
      --manifest-dir /tmp/tiny-vllm-manifests \
      --enforce-eager \
      > vllm-eager.csv

Eager mode disables CUDA Graphs and `torch.compile`, so the difference from default vLLM mode is not attributable to CUDA Graphs alone. The script disables the optional FlashInfer sampler by default to avoid JIT compilation failures when the system `nvcc` does not match the CUDA version used for the installed PyTorch/vLLM wheels.

### Measured Results

Measured on an RTX 3090 (24 GB) with Qwen2.5-3B-Instruct, FP16 runtime, batch sizes 1/8/32, 128-token prompts, and 128 generated tokens per request. All runs used random sampling, EOS ignored, 3 warmups, and 5 measured repeats; table values are median aggregate output tokens/s. TinyLLM and both vLLM modes used the same per-repeat prompt manifests. The vLLM eager result disables both CUDA Graphs and `torch.compile`.

| Batch size | TinyLLM | vLLM eager | vLLM default |
| ---: | ---: | ---: | ---: |
| 1 | 85.79 | 68.93 | 113.29 |
| 8 | 560.61 | 494.47 | 725.48 |
| 32 | 1482.61 | 1619.73 | 2183.71 |


These are measured results for this hardware, model, and workload, not a general performance guarantee. CUDA Graphs and compilation make a substantial difference to vLLM's decode throughput in these tests.

<!-- ## Model layout

- model/src/qwen3_checkpoint.cpp reads Qwen2/Qwen3 configuration and maps single-file or sharded Safetensors.
- model/src/qwen3_layers.cpp converts checkpoint tensors and builds the embedding, decoder layers, final norm, and LM head.
- model/src/qwen3.cpp handles model initialization and KV sizing.
- model/src/qwen3_kv_cache.cpp allocates the paged KV pool.
- model/src/qwen3_paged_batch.cpp runs batched Qwen2/Qwen3 tokens through the decoder and paged attention.

Each Qwen3DecoderLayer keeps its attention projections, MLP projections, and norms together.

## KV cache sizing

The Qwen3 engine loads model weights, then allocates one fixed K/V page pool before serving requests. Requests allocate and release page IDs through the block manager; GPU cache memory is not allocated per token.

For the current FP16 cache, the pool size is:

    2 * num_layers * num_blocks * block_size * kv_dim * sizeof(uint16_t)

On CUDA, the CLI estimates num_blocks from the free VRAM after loading the model. --gpu-memory-utilization controls the target fraction of total GPU memory and defaults to 0.9; the estimator leaves additional room for workspaces. Set --num-kv-blocks N to choose a fixed pool size.

On CPU, the default pool is sized to hold one sequence up to --max-seq-len. -->

