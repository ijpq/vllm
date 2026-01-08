# test_triton_only.py
import torch
from triton_kernels.routing import routing as triton_routing

torch.manual_seed(42)
router_logits = torch.randn((16, 32), dtype=torch.bfloat16, device='cuda')
result = triton_routing(router_logits, 4, sm_first=False)
print("triton_routing alone: OK")