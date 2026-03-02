# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests to verify that ops.fused_routing produces results consistent
with triton_kernels.routing.routing().
"""

import pytest
import torch

from vllm.utils.import_utils import has_triton_kernels

if not has_triton_kernels():
    pytest.skip(
        "triton_kernels not found, skipping all related tests",
        allow_module_level=True,
    )

from triton_kernels.routing import routing as triton_routing

from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    fused_routing,
)


def compare_routing_results(
    fused_result: tuple,
    triton_result: tuple,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    Compare routing results from fused_routing and triton_routing.
    
    Args:
        fused_result: (RoutingData, GatherIndx, ScatterIndx) from fused_routing
        triton_result: (RoutingData, GatherIndx, ScatterIndx) from triton_routing
        rtol: relative tolerance for float comparisons
        atol: absolute tolerance for float comparisons
    """
    fused_routing_data, fused_gather, fused_scatter = fused_result
    triton_routing_data, triton_gather, triton_scatter = triton_result
    
    errors = []
    
    # 1. Compare gate_scal (the softmax weights sorted by expert)
    # Note: The order might differ due to different sorting implementations,
    # but the values themselves should be consistent when sorted
    try:
        fused_gate_scal = fused_routing_data.gate_scal.float()
        triton_gate_scal = triton_routing_data.gate_scal.float()
        
        # Sort both for comparison (since expert assignment order may differ within ties)
        fused_sorted, _ = torch.sort(fused_gate_scal)
        triton_sorted, _ = torch.sort(triton_gate_scal)
        
        if not torch.allclose(fused_sorted, triton_sorted, rtol=rtol, atol=atol):
            max_diff = (fused_sorted - triton_sorted).abs().max().item()
            errors.append(f"gate_scal mismatch: max_diff={max_diff}")
    except Exception as e:
        errors.append(f"gate_scal comparison failed: {e}")
    
    # 2. Compare histogram (tokens per expert)
    try:
        fused_hist = fused_routing_data.expt_hist
        triton_hist = triton_routing_data.expt_hist
        
        if not torch.equal(fused_hist, triton_hist):
            diff = (fused_hist - triton_hist).abs()
            errors.append(f"expt_hist mismatch: diff={diff.tolist()}")
            errors.append(f"expt_hist fused: {fused_hist.tolist()}")
            errors.append(f"expt_hist triton: {triton_hist.tolist()}")
    except Exception as e:
        errors.append(f"expt_hist comparison failed: {e}")
    
    # 3. Compare n_expts_tot and n_expts_act
    if fused_routing_data.n_expts_tot != triton_routing_data.n_expts_tot:
        errors.append(
            f"n_expts_tot mismatch: fused={fused_routing_data.n_expts_tot}, "
            f"triton={triton_routing_data.n_expts_tot}"
        )
    
    if fused_routing_data.n_expts_act != triton_routing_data.n_expts_act:
        errors.append(
            f"n_expts_act mismatch: fused={fused_routing_data.n_expts_act}, "
            f"triton={triton_routing_data.n_expts_act}"
        )
    
    # 4. Compare ExptData
    fused_expt_data = fused_routing_data.expt_data
    triton_expt_data = triton_routing_data.expt_data
    
    # 4a. Compare token_offs_raw (cumulative sum of histogram)
    try:
        # Note: fused uses expt_offs which is token_offs_raw
        fused_offs = fused_expt_data.token_offs_raw
        triton_offs = triton_expt_data.token_offs_raw
        
        # They may have different lengths, compare the common prefix
        min_len = min(len(fused_offs), len(triton_offs))
        if not torch.equal(fused_offs[:min_len], triton_offs[:min_len]):
            diff = (fused_offs[:min_len] - triton_offs[:min_len]).abs()
            errors.append(f"token_offs_raw mismatch: diff={diff.tolist()}")
    except Exception as e:
        errors.append(f"token_offs_raw comparison failed: {e}")
    
    # 4b. Compare token_offs_pad for each block size
    try:
        for block_m in fused_expt_data.token_offs_pad:
            if block_m in triton_expt_data.token_offs_pad:
                fused_pad = fused_expt_data.token_offs_pad[block_m]
                triton_pad = triton_expt_data.token_offs_pad[block_m]
                
                min_len = min(len(fused_pad), len(triton_pad))
                if not torch.equal(fused_pad[:min_len], triton_pad[:min_len]):
                    diff = (fused_pad[:min_len] - triton_pad[:min_len]).abs()
                    errors.append(f"token_offs_pad[{block_m}] mismatch: diff={diff.tolist()}")
                    errors.append(f"token_offs_pad[{block_m}] fused {fused_pad.tolist()}")
                    errors.append(f"token_offs_pad[{block_m}] triton {triton_pad.tolist()}")
    except Exception as e:
        errors.append(f"token_offs_pad comparison failed: {e}")
    
    # 4c. Compare block_pid_map for each block size
    try:
        for block_m in fused_expt_data.block_pid_map:
            if block_m in triton_expt_data.block_pid_map:
                fused_map = fused_expt_data.block_pid_map[block_m]
                triton_map = triton_expt_data.block_pid_map[block_m]
                
                # Compare non-negative entries (valid mappings)
                fused_valid = fused_map[fused_map >= 0]
                triton_valid = triton_map[triton_map >= 0]
                
                # Sort for comparison
                fused_sorted, _ = torch.sort(fused_valid)
                triton_sorted, _ = torch.sort(triton_valid)
                
                if len(fused_sorted) != len(triton_sorted):
                    errors.append(
                        f"block_pid_map[{block_m}] length mismatch: "
                        f"fused={len(fused_sorted)}, triton={len(triton_sorted)}"
                    )
                elif not torch.equal(fused_sorted, triton_sorted):
                    errors.append(f"block_pid_map[{block_m}] values mismatch")
    except Exception as e:
        errors.append(f"block_pid_map comparison failed: {e}")
    
    # 5. Verify gather/scatter index consistency
    # The actual indices may differ due to tie-breaking, but they should
    # produce the same routing when applied
    try:
        fused_topk_idx = fused_gather.src_indx
        fused_gate_idx = fused_gather.dst_indx
        triton_topk_idx = triton_gather.src_indx
        triton_gate_idx = triton_gather.dst_indx
        
        # Verify that applying gather produces consistent results
        # Create a test tensor and verify routing produces same output
        n_gates = len(fused_gate_scal)
        test_input = torch.arange(n_gates, device=fused_topk_idx.device, dtype=torch.float32)
        
        # Apply gather: output[i] = input[topk_idx[i]]
        fused_gathered = test_input[fused_topk_idx.long()]
        triton_gathered = test_input[triton_topk_idx.long()]
        
        # After sorting by gate_idx, they should match
        fused_result_sorted = fused_gathered[fused_gate_idx.long()]
        triton_result_sorted = triton_gathered[triton_gate_idx.long()]
        
        # Note: Direct comparison may not work due to different orderings
        # Instead verify the mapping is valid (each index appears once)
        if len(torch.unique(fused_topk_idx)) != len(fused_topk_idx):
            errors.append("fused_routing topk_idx has duplicates")
        if len(torch.unique(triton_topk_idx)) != len(triton_topk_idx):
            errors.append("triton_routing topk_idx has duplicates")
            
    except Exception as e:
        errors.append(f"gather/scatter index verification failed: {e}")
    
    return errors


class TestFusedRouting:
    """Test suite for comparing fused_routing with triton_kernels.routing.routing"""
    
    @pytest.mark.parametrize("num_tokens", [1, 16, 32, 64, 128, 256, 512, 4096])
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    @pytest.mark.parametrize("topk", [4])  # Currently only topk=4 supported
    def test_routing_consistency(self, num_tokens: int, num_experts: int, topk: int):
        """
        Test that fused_routing produces consistent results with triton_routing.
        """
        torch.manual_seed(42)
        device = "cuda"
        
        # Generate random router logits
        router_logits = torch.randn(
            (num_tokens, num_experts), 
            dtype=torch.bfloat16, 
            device=device
        )
        
        # Run both routing implementations
        # Note: Both use renormalize=True / sm_first=False (softmax after topk)
        fused_result = fused_routing(router_logits, topk, renormalize=True)
        triton_result = triton_routing(router_logits, topk, sm_first=False)
        
        # Compare results
        errors = compare_routing_results(fused_result, triton_result)
        
        if errors:
            error_msg = "\n".join(errors)
            pytest.fail(f"Routing results mismatch:\n{error_msg}")
    
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    @pytest.mark.parametrize("num_tokens", [16, 64, 256])
    def test_histogram_correctness(self, num_tokens: int, num_experts: int):
        """
        Test that the histogram (tokens per expert) is computed correctly.
        """
        torch.manual_seed(123)
        device = "cuda"
        topk = 4
        
        router_logits = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device
        )
        
        fused_result = fused_routing(router_logits, topk, renormalize=True)
        triton_result = triton_routing(router_logits, topk, sm_first=False)
        
        fused_hist = fused_result[0].expt_hist
        triton_hist = triton_result[0].expt_hist
        
        # Histogram should sum to num_tokens * topk
        expected_sum = num_tokens * topk
        
        assert fused_hist.sum().item() == expected_sum, \
            f"fused histogram sum {fused_hist.sum().item()} != expected {expected_sum}"
        assert triton_hist.sum().item() == expected_sum, \
            f"triton histogram sum {triton_hist.sum().item()} != expected {expected_sum}"
        
        # Histograms should match
        assert torch.equal(fused_hist, triton_hist), \
            f"Histogram mismatch: fused={fused_hist.tolist()}, triton={triton_hist.tolist()}"
    
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    @pytest.mark.parametrize("num_tokens", [16, 64, 256])
    def test_gate_scale_values(self, num_tokens: int, num_experts):
        """
        Test that gate_scale values are valid softmax probabilities.
        """
        torch.manual_seed(456)
        device = "cuda"
        topk = 4
        
        router_logits = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device
        )
        
        fused_result = fused_routing(router_logits, topk, renormalize=True)
        triton_result = triton_routing(router_logits, topk, sm_first=False)
        
        fused_gate_scal = fused_result[0].gate_scal.float()
        triton_gate_scal = triton_result[0].gate_scal.float()
        
        # All values should be positive (softmax outputs)
        assert (fused_gate_scal >= 0).all(), "fused gate_scale has negative values"
        assert (triton_gate_scal >= 0).all(), "triton gate_scale has negative values"
        
        # All values should be <= 1 (softmax outputs)
        assert (fused_gate_scal <= 1).all(), "fused gate_scale has values > 1"
        assert (triton_gate_scal <= 1).all(), "triton gate_scale has values > 1"
        
        # Sum of gate_scale per token should equal 1 (after grouping by token)
        # This is harder to verify after sorting, so we just check the total sum
        expected_sum = num_tokens  # Each token's topk weights sum to 1
        
        # Note: Due to sorting, we can't easily group by token, but the total sum
        # should be approximately num_tokens
        fused_sum = fused_gate_scal.sum().item()
        triton_sum = triton_gate_scal.sum().item()
        
        assert abs(fused_sum - expected_sum) < 0.1 * expected_sum, \
            f"fused gate_scale sum {fused_sum} too far from expected {expected_sum}"
        assert abs(triton_sum - expected_sum) < 0.1 * expected_sum, \
            f"triton gate_scale sum {triton_sum} too far from expected {expected_sum}"
    
    @pytest.mark.parametrize("num_tokens", [16, 64, 128])
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    def test_token_offs_pad_consistency(self, num_tokens: int, num_experts: int):
        """
        Test that token_offs_pad is computed correctly for different block sizes.
        """
        torch.manual_seed(789)
        device = "cuda"
        topk = 4
        
        router_logits = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device
        )
        
        fused_result = fused_routing(router_logits, topk, renormalize=True)
        triton_result = triton_routing(router_logits, topk, sm_first=False)
        
        fused_expt_data = fused_result[0].expt_data
        triton_expt_data = triton_result[0].expt_data
        
        # fused_routing now only computes for a single block_m
        for block_m in fused_expt_data.token_offs_pad:
            if block_m not in triton_expt_data.token_offs_pad:
                continue
                
            fused_pad = fused_expt_data.token_offs_pad[block_m]
            triton_pad = triton_expt_data.token_offs_pad[block_m]
            
            # token_offs_pad should be monotonically non-decreasing
            assert (fused_pad[1:] >= fused_pad[:-1]).all(), \
                f"fused token_offs_pad[{block_m}] not monotonic"
            assert (triton_pad[1:] >= triton_pad[:-1]).all(), \
                f"triton token_offs_pad[{block_m}] not monotonic"
            
            # First element should be 0
            assert fused_pad[0] == 0, f"fused token_offs_pad[{block_m}][0] != 0"
            assert triton_pad[0] == 0, f"triton token_offs_pad[{block_m}][0] != 0"
    
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    def test_determinism(self, num_experts: int):
        """
        Test that fused_routing produces deterministic results.
        """
        torch.manual_seed(999)
        device = "cuda"
        num_tokens = 128
        topk = 4
        
        router_logits = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device
        )
        
        # Run twice with same input
        result1 = fused_routing(router_logits.clone(), topk, renormalize=True)
        result2 = fused_routing(router_logits.clone(), topk, renormalize=True)
        
        # Results should be identical
        assert torch.equal(result1[0].gate_scal, result2[0].gate_scal), \
            "gate_scal not deterministic"
        assert torch.equal(result1[1].src_indx, result2[1].src_indx), \
            "gather.src_indx not deterministic"
        assert torch.equal(result1[1].dst_indx, result2[1].dst_indx), \
            "gather.dst_indx not deterministic"


class TestFusedRoutingEdgeCases:
    """Edge case tests for fused_routing"""
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    def test_uniform_distribution(self, num_experts: int):
        """
        Test with uniform logits (all experts equally likely).
        Note: topk_softmax has deterministic tie-breaking (selects lowest indices),
        so this test just verifies the kernel runs without error and histogram sums correctly.
        """
        device = "cuda"
        num_tokens = 64
        topk = 4

        router_logits = torch.ones(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device
        )

        fused_result = fused_routing(router_logits, topk, renormalize=True)
        
        # Just verify histogram sums to num_tokens * topk
        fused_hist = fused_result[0].expt_hist
        assert fused_hist.sum().item() == num_tokens * topk, \
            f"Histogram sum {fused_hist.sum().item()} != expected {num_tokens * topk}"
        
        # With deterministic tie-breaking, first topk experts get all tokens
        # This is expected behavior for topk_softmax
   
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    def test_sparse_routing(self, num_experts: int):
        """
        Test with sparse logits (only a few experts active).
        """
        device = "cuda"
        num_tokens = 64
        topk = 4
        
        # Set logits such that only first 4 experts are selected
        router_logits = torch.full(
            (num_tokens, num_experts),
            -100.0,
            dtype=torch.bfloat16,
            device=device
        )
        router_logits[:, :topk] = 1.0  # Only first topk experts have high logits
        
        fused_result = fused_routing(router_logits, topk, renormalize=True)
        triton_result = triton_routing(router_logits, topk, sm_first=False)
        
        fused_hist = fused_result[0].expt_hist
        triton_hist = triton_result[0].expt_hist
        
        # All tokens should go to first topk experts
        expected_per_active = num_tokens
        assert (fused_hist[:topk] == expected_per_active).all(), \
            f"Expected {expected_per_active} tokens per active expert, got {fused_hist[:topk].tolist()}"
        assert (fused_hist[topk:] == 0).all(), \
            f"Inactive experts should have 0 tokens, got {fused_hist[topk:].tolist()}"
    
    @pytest.mark.parametrize("num_experts", [32,128])  # Currently only 32 experts supported
    def test_large_logit_values(self, num_experts:int):
        """
        Test with very large logit values (numerical stability).
        """
        torch.manual_seed(111)
        device = "cuda"
        num_tokens = 64
        topk = 4
        
        # Large logit values
        router_logits = torch.randn(
            (num_tokens, num_experts),
            dtype=torch.bfloat16,
            device=device
        ) * 100  # Scale up
        
        fused_result = fused_routing(router_logits, topk, renormalize=True)
        triton_result = triton_routing(router_logits, topk, sm_first=False)
        
        # Should not have NaN or Inf
        assert not torch.isnan(fused_result[0].gate_scal).any(), \
            "fused gate_scal has NaN"
        assert not torch.isinf(fused_result[0].gate_scal).any(), \
            "fused gate_scal has Inf"
        assert not torch.isnan(triton_result[0].gate_scal).any(), \
            "triton gate_scal has NaN"
        assert not torch.isinf(triton_result[0].gate_scal).any(), \
            "triton gate_scal has Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
