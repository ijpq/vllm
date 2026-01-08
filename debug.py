# test_debug_v2.py
import torch

from vllm.utils.import_utils import import_triton_kernels
import_triton_kernels()
from triton_kernels.routing import routing as triton_routing

try:
    torch.manual_seed(42)
    router_logits = torch.randn((32, 32), dtype=torch.bfloat16, device='cuda')
    result = triton_routing(router_logits, 4, sm_first=False)
    print("triton_routing after fused: OK")
except Exception as e:
    print(f"triton_routing after fused FAILED: {e}")
torch.cuda.synchronize()

# Test fused_routing alone
print("\nTesting fused_routing...")
try:
    from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import fused_routing
    torch.manual_seed(42)
    router_logits = torch.randn((32, 32), dtype=torch.bfloat16, device='cuda')
    result = fused_routing(router_logits, 4, renormalize=True)
    torch.cuda.synchronize()
    print("fused_routing: OK")
except Exception as e:
    print(f"fused_routing FAILED: {e}")
