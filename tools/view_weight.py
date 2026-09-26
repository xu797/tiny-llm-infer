from safetensors import safe_open
import torch
# ===================== 改这里！换成你的safetensors文件路径 =====================
PATH = "/home/eric/xujixian/work/tiny-llm-infer/Qwen3-0.6B/model.safetensors"
# ==============================================================================


with safe_open(PATH, framework="pt", device="cpu") as f:
    print(f"✅ 一共 {len(f.keys())} 个张量\n")
    for name in f.keys():
        tensor = f.get_tensor(name)
        print(f"tensor name: {name}")
        print(f"  shape: {tensor.shape}")
        print(f"  dtype: {tensor.dtype}")
        print(f"  numel: {tensor.numel()}")
        print("-" * 60)
