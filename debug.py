# test_debug_v2.py
import torch

# 正确的导入方式 - 先调用 import_triton_kernels() 注册模块
from vllm.utils.import_utils import import_triton_kernels
import_triton_kernels()

# Test triton_routing alone (fresh CUDA context)
print("Testing triton_routing alone (fresh context)...")
try:
    from triton_kernels.routing import routing as triton_routing
    torch.manual_seed(42)
    router_logits = torch.randn((16, 32), dtype=torch.bfloat16, device='cuda')
    result = triton_routing(router_logits, 4, sm_first=False)
    print("triton_routing: OK")
except Exception as e:
    print(f"triton_routing FAILED: {e}")

# Test fused_routing alone
print("\nTesting fused_routing...")
try:
    from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import fused_routing
    torch.manual_seed(42)
    router_logits = torch.randn((32, 32), dtype=torch.bfloat16, device='cuda')
    result = fused_routing(router_logits, 4, renormalize=True)
    print("fused_routing: OK")
except Exception as e:
    print(f"fused_routing FAILED: {e}")

# Test triton_routing AFTER fused_routing (this is where it was failing)
print("\nTesting triton_routing AFTER fused_routing...")
try:
    torch.manual_seed(42)
    router_logits = torch.randn((32, 32), dtype=torch.bfloat16, device='cuda')
    result = triton_routing(router_logits, 4, sm_first=False)
    print("triton_routing after fused: OK")
except Exception as e:
    print(f"triton_routing after fused FAILED: {e}")