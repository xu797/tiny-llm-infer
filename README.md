# tiny-llm-infer

A small C++ inference engine with Qwen3 prompt batching and a preallocated paged KV cache.

## Build

Run from the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j2

## Qwen3 inference

Single request:

    ./build/model/model_main --infer \
      --tokenizer Qwen3-0.6B/tokenizer.json \
      --model Qwen3-0.6B/model.safetensors \
      --prompt $'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n' \
      --device cuda --max-seq-len 2048 --max-new-tokens 64

Pass --prompt more than once to submit a batch:

    ./build/model/model_main --infer \
      --tokenizer Qwen3-0.6B/tokenizer.json \
      --model Qwen3-0.6B/model.safetensors \
      --prompt 'hello' --prompt 'introduce qwen3 model.' \
      --device cuda --max-seq-len 2048 --max-new-tokens 64

The CLI treats each prompt as text to continue; it does not apply tokenizer_config.json's chat template automatically. For instruction-style replies, wrap each user message with Qwen3 chat markers as shown above. Each sequence decodes one new token per step. Active sequences are decoded together. Prompt prefill batches multiple tokens, and long prompts can be split across multiple prefill steps according to the scheduler token budget.

## Qwen3 model layout

- model/src/qwen3_checkpoint.cpp reads model configuration and maps Safetensors.
- model/src/qwen3_layers.cpp converts checkpoint tensors and builds the embedding, decoder layers, final norm, and LM head.
- model/src/qwen3.cpp handles model initialization and KV sizing.
- model/src/qwen3_kv_cache.cpp allocates the paged KV pool.
- model/src/qwen3_paged_batch.cpp runs batched Qwen3 tokens through the decoder and paged attention.

Each Qwen3DecoderLayer keeps its attention projections, MLP projections, and norms together.

## KV cache sizing

The Qwen3 engine loads model weights, then allocates one fixed K/V page pool before serving requests. Requests allocate and release page IDs through the block manager; GPU cache memory is not allocated per token.

For the current FP32 cache, the pool size is:

    2 * num_layers * num_blocks * block_size * kv_dim * sizeof(float)

On CUDA, the CLI estimates num_blocks from the free VRAM after loading the model. --gpu-memory-utilization controls the target fraction of total GPU memory and defaults to 0.9; the estimator leaves additional room for workspaces. Set --num-kv-blocks N to choose a fixed pool size.

On CPU, the default pool is sized to hold one sequence up to --max-seq-len.

cmake --build build -j2

./build/model/model_main --infer \
  --tokenizer Qwen3-0.6B/tokenizer.json \
  --model Qwen3-0.6B/model.safetensors \
  --prompt $'<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n' \
  --prompt $'<|im_start|>user\nintroduce qwen3 model.<|im_end|>\n<|im_start|>assistant\n' \
  --device cuda --max-seq-len 2048 --max-new-tokens 1024

