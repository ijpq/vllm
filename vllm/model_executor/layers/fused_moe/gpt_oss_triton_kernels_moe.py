# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import vllm_topk_softmax
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton.topk import topk_forward
        from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs
        from triton_kernels.routing import (
            ExptData,
            GatherIndx,
            RoutingData,
            ScatterIndx,
            routing_from_bitmatrix,
        )
        from triton_kernels.tensor import Bitmatrix
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)



@triton.jit
def _cdiv(n, d):
    return (n + d - 1) // d


def fused_routing(
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple["RoutingData", "GatherIndx", "ScatterIndx"]:
    """
    Fully fused topk, softmax, routing, and ExptData computation.
    """
    M, N = router_logits.shape
    device = router_logits.device
    dtype = router_logits.dtype

    topk_weights = torch.empty((M, topk), device=device, dtype=dtype)
    topk_indices = torch.empty((M, topk), device=device, dtype=torch.int16)
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=router_logits.device
    )

    hist = torch.zeros(N, device=device, dtype=torch.int32)
    expt_offs = torch.empty(N + 1, device=device, dtype=torch.int32)

    n_gates = M * topk
    gate_scale = torch.empty(n_gates, device=device, dtype=dtype)
    topk_index = torch.empty(n_gates, device=device, dtype=torch.int32)
    gate_index = torch.empty(n_gates, device=device, dtype=torch.int32)

    # block_m sizes: 16, 32, 64, 128 (NUM_BLOCK_SIZES=4)
    BLOCK_M_LOG2_START = 4
    NUM_BLOCK_SIZES = 4

    if n_gates <= N:
        max_n_tiles = n_gates
    else:
        min_block_m = 1 << BLOCK_M_LOG2_START  # 16
        max_n_tiles = N - 1 - ((N - n_gates - 1) // min_block_m)

    token_offs_pad = torch.empty(
        (NUM_BLOCK_SIZES, N + 1), device=device, dtype=torch.int32
    )
    block_pid_map = torch.full(
        (NUM_BLOCK_SIZES, max_n_tiles), -1, device=device, dtype=torch.int32
    )


    # device_props = torch.cuda.get_device_properties(device)
    # num_sms = device_props.multi_processor_count

    # max_num_programs = 64
    # ROWS_PER_PID = 4
    # desired_programs = triton.cdiv(M, ROWS_PER_PID)
    # num_programs = min(desired_programs, num_sms, max_num_programs)
    # ROWS_PER_PID = triton.cdiv(M, num_programs)
    # hist = torch.zeros((ROWS_PER_PID,N), device=device, dtype=torch.int32)

    # partial_hist = torch.zeros((num_programs, N), device=device, dtype=torch.int32)

    # XXX: Since we have fused topk+softmax kernel, leave it outside
    topk_weights, topk_indices = vllm_topk_softmax(
        topk_weights, topk_indices, token_expert_indices, router_logits, renormalize
    )

    ops.fused_routing(
        router_logits,
        topk_weights,
        topk_indices,
        max_n_tiles,
        topk,
        renormalize,
        gate_scale,
        topk_index,
        gate_index,
        token_offs_pad,
        block_pid_map,
        expt_offs
    )

    gate_scale = gate_scale.to(torch.bfloat16)
    topk_weights = topk_weights.to(torch.bfloat16)

    token_offs_pad_dict = {
        (1 << (BLOCK_M_LOG2_START + i)): token_offs_pad[i]
        for i in range(NUM_BLOCK_SIZES)
    }
    block_pid_map_dict = {
        (1 << (BLOCK_M_LOG2_START + i)): block_pid_map[i]
        for i in range(NUM_BLOCK_SIZES)
    }

    # token_offs_raw = expt_offs[: N + 1]

    expt_data = ExptData(
        hist=hist,
        token_offs_raw=expt_offs,
        token_offs_pad=token_offs_pad_dict,
        block_pid_map=block_pid_map_dict,
    )

    gather_index = GatherIndx(topk_index, gate_index)
    scatter_index = ScatterIndx(gate_index, topk_index)
    routing_data = RoutingData(
        gate_scal=gate_scale,
        expt_hist=hist,
        n_expts_tot=N,
        n_expts_act=topk,
        expt_data=expt_data,
    )
    return routing_data, gather_index, scatter_index


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = fused_routing(
        gating_output, topk, renormalize
    )

    output = torch.empty_like(hidden_states)

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, N // 2)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )

    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    output_tensor = output_tensor.view(M, K)
    return output_tensor


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = Bitmatrix(
        bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=None
    )

    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        super().__init__(quant_config)

    def supports_expert_map(self) -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert w1.dim() == 3 and w2.dim() == 3
        E, _, N = w1.size()
        K = a1.size(-1)

        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
        return make_routing_data(topk_ids, topk_weights, num_local_experts)


class OAITritonExperts(BaseOAITritonExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0, 0)
        workspace2 = (M * topk, N // 2)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)
        triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            topk=topk,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=local_num_experts,
            expert_map=None,
            intermediate_cache=workspace2,
            a1q_scale=a1q_scale,
        )


class UnfusedOAITritonExperts(BaseOAITritonExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, N // 2)
        workspace2 = (M * topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor):
        ops.moe_sum(input, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.quant_config is None:
            self.quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        topk = topk_ids.size(1)

        assert hidden_states.dtype == torch.bfloat16
        assert (
            self.quant_config.w1_bias is None
            or self.quant_config.w1_bias.dtype == torch.float32
        )
        assert (
            self.quant_config.w2_bias is None
            or self.quant_config.w2_bias.dtype == torch.float32
        )

        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

        batch_dim = 1
        M, K = hidden_states.shape
        E, _, N = w1.shape

        if global_num_experts == -1:
            global_num_experts = E

        intermediate_cache1 = _resize_cache(workspace2, (batch_dim, M * topk, N))
        intermediate_cache3 = _resize_cache(workspace2, (batch_dim, M * topk, K))
        intermediate_cache2 = _resize_cache(workspace13, (M * topk, N // 2))

        gammas = routing_data.gate_scal if routing_data else None

        matmul_ogs(
            hidden_states,
            w1,
            self.quant_config.w1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=self.quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=None,
            y=intermediate_cache1,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        routing_data.n_expts_act = 1

        matmul_ogs(
            intermediate_cache2,
            w2,
            self.quant_config.w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=self.quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
            y=intermediate_cache3,
        )

        self.moe_sum(intermediate_cache3.view(-1, topk, K), output)
