# tiny-llm-infer
...

```
Build from the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j2

infer:
./build/model/model_main --infer --model-type qwen3   --tokenizer Qwen3-0.6B/tokenizer.json   --model Qwen3-0.6B/model.safetensors   --prompt $'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n'   --device cuda --max-seq-len 2048 --max-new-tokens 64
```

Qwen3 accepts repeated --prompt options. The engine batches prompt tokens and decode requests, and stores KV states in reusable fixed-size pages:

    ./build/model/model_main --infer --model-type qwen3 --tokenizer Qwen3-0.6B/tokenizer.json --model Qwen3-0.6B/model.safetensors --prompt 'First prompt' --prompt 'Second prompt' --device cuda --max-seq-len 2048 --max-new-tokens 64

The C++ engine API is in engine/include/llm_engine.h. EngineConfig::scheduler controls the active sequence limit, prefill token budget, KV page size, and page count. The CLI routes Qwen3 through this engine; the legacy Llama2 path still accepts one prompt.


```
./build/model/model_main --infer \
  --model-type qwen3 \
  --tokenizer Qwen3-0.6B/tokenizer.json \
  --model Qwen3-0.6B/model.safetensors \
  --prompt $'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n' \
  --prompt $'<|im_start|>user\n请简要介绍 paged KV cache。<|im_end|>\n<|im_start|>assistant\n' \
  --device cuda \
  --max-seq-len 2048 \
  --max-new-tokens 1024
```