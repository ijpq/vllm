import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查 _moe_C 模块是否存在
try:
    from vllm import _custom_ops as ops
    print(f"Has topk_softmax: {hasattr(torch.ops._moe_C, 'topk_softmax')}")
except Exception as e:
    print(f"Error: {e}")
