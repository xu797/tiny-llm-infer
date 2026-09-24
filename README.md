# tiny-llm-infer
...

```
infer:
./model/model_main --infer --model-type qwen3   --tokenizer ../Qwen3-0.6B/tokenizer.json   --model ../Qwen3-0.6B/model.safetensors   --prompt $'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n'   --device cuda --max-seq-len 2048 --max-new-tokens 64
```


