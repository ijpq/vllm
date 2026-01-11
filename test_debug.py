#!/usr/bin/env python3
"""Debug test for fused_routing kernel"""
import torch

# Reproduce the issue with a simple test
def test_fused_routing():
    from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import fused_routing
    
    torch.manual_seed(42)
    device = "cuda"
    
    # Test with small batch first
    for num_tokens in [16, 64, 128, 256, 512, 1024]:
        print(f"\n=== Testing num_tokens={num_tokens} ===")
        num_experts = 32
        topk = 4
        
        router_logits = torch.randn(
            (num_tokens, num_experts), 
            dtype=torch.bfloat16, 
            device=device
        )
        
        try:
            result = fused_routing(router_logits, topk, renormalize=True)
            print(f"  SUCCESS: gate_scal shape = {result[0].gate_scal.shape}")
            
            # Verify histogram
            hist = result[0].expt_hist
            expected_sum = num_tokens * topk
            actual_sum = hist.sum().item()
            print(f"  Histogram sum: {actual_sum} (expected {expected_sum})")
            
            if actual_sum != expected_sum:
                print(f"  ERROR: Histogram sum mismatch!")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    test_fused_routing()
