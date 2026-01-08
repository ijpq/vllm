#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"
#include "../cub_helpers.h"
#include <cooperative_groups.h>

#ifndef USE_ROCM
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
#else
    #include <hip/hip_bf16.h>
    #include <hip/hip_fp16.h>
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#endif
namespace vllm {
namespace moe {

// IndexType <- int16*
template <int NUM_EXPERTS, int topk, int NUM_BLOCK_SIZES>
__global__ void fused_routing_kernel(
    __nv_bfloat16* __restrict__ topk_weights,
    int16_t* __restrict__ topk_indices, const int64_t max_n_tiles,
    const int64_t NUM_TOKENS, __nv_bfloat16* __restrict__ gate_scale,
    int32_t* __restrict__ topk_index, int32_t* __restrict__ gate_index,
    int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr);

// template<>
// __global__
// void fused_routing_kernel<128>( float* __restrict__ router_logits,
//                            float* __restrict__ topk_weights,
//                            int16_t* __restrict__ topk_indices,
//                            float* __restrict__ gate_scale,
//                            int16_t* __restrict__ topk_index,
//                            int32_t* __restrict__ gate_index,
//                           int64_t max_n_tiles,  int64_t topk) {
// }

template <>
__global__ void fused_routing_kernel<32, 4, 4>(
    __nv_bfloat16* __restrict__ topk_weights,
    int16_t* __restrict__ topk_indices, const int64_t max_n_tiles,
    const int64_t NUM_TOKENS, __nv_bfloat16* __restrict__ gate_scale,
    int32_t* __restrict__ topk_index, int32_t* __restrict__ gate_index,
    int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr) {
    // static_assert(alignof(topk_indices) == 128);
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int local_tid = threadIdx.x;
    int tid = local_tid + blockDim.x * blockIdx.x;
    int CTA_ID = blockIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    const int NUM_EXPERTS = 32;
    const int topk = 4;
    constexpr int NUM_BLOCK_SIZES = 4;
    int ROWS_PER_THREADS = (NUM_TOKENS + num_threads - 1) / num_threads;
    int ROWS_PER_CTA = (NUM_TOKENS + gridDim.x - 1) / gridDim.x;

    // shared memory layout
    /*
    Assumingly, shared Mem is organized as this layout:
    [NUM_EXPERTS] : global hist !only 0
    [NUM_EXPERTS] : local histgram for each threadblock
    [NUM_EXPERTS+1]: global hist exclusive-sum !only 0
    [NUM_BLOCK_SIZES, NUM_EXPERTS+1] : token_offs_pad
    [NUM_BLOCK_SIZES, max_n_tiles] : block_pid
    [NUM_EXPERTS]: exclusivesum for experts
    [ROWS_PER_CTA*topk]: local_offset
    */
    using WarpScan = cub::WarpScan<int>;
    __shared__ typename WarpScan::TempStorage temp_storage[4];
    extern __shared__ int32_t sm_hist[];
    int global_hist_offset = 0;
    int local_hist_offset = NUM_EXPERTS;
    int global_hist_exclusivesum_offset = local_hist_offset + NUM_EXPERTS;
    int token_offs_pad_offset =
        global_hist_exclusivesum_offset + NUM_EXPERTS + 1;
    int block_pid_offset =
        token_offs_pad_offset + (NUM_BLOCK_SIZES * (NUM_EXPERTS + 1));
    int expert_across_offset =
        block_pid_offset + (NUM_BLOCK_SIZES * max_n_tiles);
    int local_offset_offset = expert_across_offset + NUM_EXPERTS;
    int shared_mem_size = local_offset_offset + topk * ROWS_PER_CTA;
#pragma unroll
    for (int i = local_tid; i < shared_mem_size; i += blockDim.x) {
        sm_hist[i] = 0;
    }
    cluster.sync();  // we need to ensure global hist in CTA0 had been memset.

    /*phase 1*/
    int32_t* local_hist =
        reinterpret_cast<int32_t*>(sm_hist + local_hist_offset);
    int32_t* local_offset_sm =
        reinterpret_cast<int32_t*>(sm_hist + local_offset_offset);
    int row = CTA_ID * ROWS_PER_CTA;
    int row_end = min((int64_t)(row + ROWS_PER_CTA),
                      NUM_TOKENS);  // Each CTA only processes its own rows
#pragma unroll
    for (int i = row + local_tid; i < row_end;
         i += blockDim.x) {  // mem transaction = warp_size * topk *
                             // sizeof(topk_indices)
        // int64_t row_experts =
        //     *reinterpret_cast<int64_t*>(const_cast<int16_t*>(topk_indices) +
        //     i * topk);
        int64_t row_experts =
            *reinterpret_cast<int64_t*>((topk_indices) + i * topk);
        auto expt0 = static_cast<int32_t>(row_experts & 0xFFFF);
        auto expt1 = static_cast<int32_t>(row_experts >> 16 & 0xFFFF);
        auto expt2 = static_cast<int32_t>(row_experts >> 32 & 0xFFFF);
        auto expt3 = static_cast<int32_t>(row_experts >> 48 & 0xFFFF);
        int local_i = i - row;
        local_offset_sm[local_i * topk] = atomicAdd(local_hist + expt0, 1);
        local_offset_sm[local_i * topk + 1] = atomicAdd(local_hist + expt1, 1);
        local_offset_sm[local_i * topk + 2] = atomicAdd(local_hist + expt2, 1);
        local_offset_sm[local_i * topk + 3] = atomicAdd(local_hist + expt3, 1);
    }
    __syncthreads();
    int32_t* global_hist = cluster.map_shared_rank(
        reinterpret_cast<int32_t*>(sm_hist + global_hist_offset), 0);
    if (local_tid < NUM_EXPERTS) {
        atomicAdd(global_hist + local_tid, local_hist[local_tid]);
    }
    cluster.sync();

    /* phase 2*/
    int warp_id = threadIdx.x / 32;
    // compute expert_across_prefixsum, dst_experts[i] =
    // sum_{p=0}^{j-1}{local_hist[p]}, where i is expert_id, j is CTA_ID
    // TODO(ijpq): we can leverage warpscan to improve this process, but need
    // assign warp threads into [NUM_CTAS, NUM_EXPERTS] carefully.
    if (CTA_ID == 0 && local_tid < NUM_EXPERTS) {
        for (int bidx = 1; bidx < gridDim.x; bidx++) {
            int32_t* dst_experts = cluster.map_shared_rank(
                reinterpret_cast<int32_t*>(sm_hist + expert_across_offset),
                bidx);
            int32_t* tb_local_hist = cluster.map_shared_rank(
                reinterpret_cast<int32_t*>(sm_hist + local_hist_offset),
                bidx - 1);
            int32_t* accum_hist = cluster.map_shared_rank(
                reinterpret_cast<int32_t*>(sm_hist + expert_across_offset),
                bidx - 1);
            dst_experts[local_tid] =
                tb_local_hist[local_tid] + accum_hist[local_tid];
        }
    }
    if (CTA_ID == 0 && warp_id < NUM_BLOCK_SIZES) {
        int lane_id = threadIdx.x % 32;
        int h =
            global_hist[lane_id];  // how many tokens are routed to this
                                   // expert, globally. let's say [100,20,50]
        if (local_tid < NUM_EXPERTS) {
            int exclusive_res = 0;
            int warp_reduce = 0;
            WarpScan(temp_storage[warp_id])
                .ExclusiveSum(h, exclusive_res, warp_reduce);
            __syncwarp();  // to ensure exclusive sum finished within one warp
            int32_t* hist_sum = reinterpret_cast<int32_t*>(
                sm_hist + global_hist_exclusivesum_offset);
            hist_sum[local_tid] = exclusive_res;
            if (local_tid == 0) hist_sum[NUM_EXPERTS] = warp_reduce;
        }

        // compute global hist exclusive sum
        // we get exclusive sum  [0, 100,120, 170]

        // align the data with triton's matmul_ogs
        int BLOCK_M_LOG2_START = 4;
        int size = warp_id;
        int block_m_log2 = BLOCK_M_LOG2_START + size;
        int block_m = 1 << block_m_log2;  // block_m = 16, 32,64,128
        int32_t* pid_map_row =
            reinterpret_cast<int32_t*>(sm_hist + block_pid_offset) +
            size * max_n_tiles;

        int n_tiles =
            (h + block_m - 1) / block_m;  // group num_tokens into tiles, [7, 1,
                                          // 8] when block_m = 16
        int warp_reduce = 0;
        int exclusive_res = 0;
        WarpScan(temp_storage[warp_id])
            .ExclusiveSum(n_tiles, exclusive_res, warp_reduce);
        __syncwarp();
        // we get exclusive sum  [0, 7, 8, 16]

        // compute tiles exclusive sum
        int32_t* token_offs_pad =
            reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
        token_offs_pad[size * (NUM_EXPERTS + 1) + lane_id] = exclusive_res;
        if (lane_id == 0)
            token_offs_pad[size * (NUM_EXPERTS + 1) + 32] = warp_reduce;

        int tile_start = token_offs_pad[size * (NUM_EXPERTS + 1) + lane_id];
        /*
        tid0: h = 100, n_tiles = 7; tid1: h=20, n_tiles= 1; tid2: h = 50,
        n_tiles=8;

        tid0:
        0<<16 | 0;
        1<<16 | 0;
        ...
        6<<16 | 0;

        tid1:
        0<<16 | 1;

        tid2:
        0<<16 | 2;
        1<<16 | 2;
        ...
        7<<16 | 2;
        */
        for (int block_idx = 0; block_idx < n_tiles; block_idx++) {
            int packed_val = (block_idx << 16) | lane_id;
            pid_map_row[(tile_start + block_idx)] = packed_val;
        }
    }
    cluster.sync();

    // WB global memory(pidmap, tokenoffspad, global hist sum)
    int token_offs_pad_size = NUM_BLOCK_SIZES * (NUM_EXPERTS + 1);
    int pid_map_size = NUM_BLOCK_SIZES * (max_n_tiles);
    // int32_t* token_offs_pad = cluster.map_shared_rank(
    //     reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset), 0);
    int32_t* token_offs_pad =
        reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
    // int32_t* block_pid = cluster.map_shared_rank(
    //     reinterpret_cast<int32_t*>(sm_hist + block_pid_offset), 0);
    int32_t* block_pid = reinterpret_cast<int32_t*>(sm_hist + block_pid_offset);

    if (CTA_ID == 0) {
        for (int i = local_tid; i < token_offs_pad_size; i += blockDim.x)
            token_offs_pad_ptr[i] = token_offs_pad[i];
        for (int i = local_tid; i < pid_map_size; i += blockDim.x)
            block_pid_map_ptr[i] = block_pid[i];
    }
    if (CTA_ID == 0 && local_tid < NUM_EXPERTS) {
        int32_t* hist_sum = reinterpret_cast<int32_t*>(
            sm_hist + global_hist_exclusivesum_offset);
        int32_t* global_hist_sm =
            reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
        expt_offs_ptr[local_tid] = hist_sum[local_tid];
        if (local_tid == 0) expt_offs_ptr[NUM_EXPERTS] = hist_sum[NUM_EXPERTS];
        hist_ptr[local_tid] = global_hist_sm[local_tid];
    }
    cluster.sync();

    /* phase 3*/

    int32_t* hist_sum = cluster.map_shared_rank(
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset),
        0);
    int32_t* prior_contrib =
        reinterpret_cast<int32_t*>(sm_hist + expert_across_offset);
    // int32_t* local_offset_sm = reinterpret_cast<int32_t*>(sm_hist + );
#pragma unroll
    for (int i = row + local_tid; i < row_end; i += blockDim.x) {
        int local_i = i - row;
        int topk_idx_stride = topk, topk_val_stride = topk;
        for (int k = 0; k < topk; k++) {
            int expert_id = topk_indices[i * topk_idx_stride + k];
            __nv_bfloat16 val = topk_weights[i * topk_val_stride + k];
            int flat_idx = i * topk + k;
            int expert_base = hist_sum[expert_id];
            int expert_prior = prior_contrib[expert_id];
            int expert_local = local_offset_sm[local_i * topk + k];
            int global_pos = expert_base + expert_prior + expert_local;

            gate_scale[global_pos] = val;
            topk_index[global_pos] = flat_idx;
            gate_index[flat_idx] = global_pos;
        }
    }
}

}  // namespace moe
}  // namespace vllm

template <typename ValType>
void routing_kernel_helper(torch::Tensor& gating_output,
                           torch::Tensor& topk_weights,
                           torch::Tensor& topk_indices, int64_t max_n_tiles,
                           int64_t topk,

                           torch::Tensor& gate_scale, torch::Tensor& topk_index,
                           torch::Tensor& gate_index,
                           torch::Tensor& token_offs_pad,
                           torch::Tensor& block_pid_map,
                           torch::Tensor& expt_offs, torch::Tensor& hist) {
    /*
    dispatch config
    */
    TORCH_CHECK(topk == 4, "");
    constexpr int const_topk = 4;
    constexpr int NUM_BLOCK_SIZES = 4;
    const auto num_experts = gating_output.size(-1);
    const auto num_tokens = gating_output.numel() / num_experts;

    switch (num_experts) {
        case 32: {
            auto warp_size = 32;
            int cluster_size = 0;
            int THREAD_PER_CTA = 512;
            cudaLaunchConfig_t config = {};
            auto kernel_wrapper =
                &(vllm::moe::fused_routing_kernel<32, const_topk,
                                                  NUM_BLOCK_SIZES>);

            int hypo_cluster_size = 8;
            size_t rows_per_cta =
                (num_tokens + hypo_cluster_size - 1) / hypo_cluster_size;
            size_t global_hist_size = num_experts;
            size_t local_hist_size = num_experts;
            size_t global_hist_prefix_size = num_experts + 1;
            size_t token_offs_pad_size = NUM_BLOCK_SIZES * (num_experts + 1);
            size_t block_pid_size = NUM_BLOCK_SIZES * (max_n_tiles);
            size_t prefix_experts_size = num_experts;
            size_t local_offset_size =
                topk * rows_per_cta;  // Use actual rows_per_cta
            config.dynamicSmemBytes =
                (global_hist_size + local_hist_size + global_hist_prefix_size +
                 token_offs_pad_size + block_pid_size + prefix_experts_size +
                 local_offset_size) *
                sizeof(int32_t);
            // config.dynamicSmemBytes = 200*1024;  // Minimal for query
            cudaFuncSetAttribute(kernel_wrapper,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed,
                                 1);
            cudaError_t err = cudaOccupancyMaxPotentialClusterSize(
                &cluster_size, kernel_wrapper, &config);
            std::cout << "cudaOccupancyMaxPotentialClusterSize returned: "
                      << cudaGetErrorString(err) << std::endl;
            std::cout << "cluster size: " << cluster_size << std::endl;

            // Now calculate shared memory with actual cluster_size
            // Use ceiling division to match kernel's ROWS_PER_CTA calculation

            std::cout << "Req Smem: " << config.dynamicSmemBytes << " bytes"
                      << std::endl;
            auto grid_dim = dim3(cluster_size, 1, 1);
            config.blockDim = dim3(THREAD_PER_CTA, 1, 1);
            config.gridDim = grid_dim;

            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x =
                grid_dim.x;  // Cluster size in X-dimension
            attribute[0].val.clusterDim.y = 1;
            attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute;
            config.numAttrs = 1;

            auto topk_weights_ptr =
                reinterpret_cast<ValType*>(topk_weights.data_ptr());
            auto gate_scale_ptr =
                reinterpret_cast<ValType*>(gate_scale.data_ptr());
            auto topk_indices_ptr = topk_indices.data_ptr<int16_t>();
            auto topk_index_ptr = topk_index.data_ptr<int32_t>();
            auto gate_index_ptr = gate_index.data_ptr<int32_t>();
            auto token_offs_pad_ptr = token_offs_pad.data_ptr<int32_t>();
            auto block_pid_map_ptr = block_pid_map.data_ptr<int32_t>();
            auto expt_offs_ptr = expt_offs.data_ptr<int32_t>();
            auto hist_ptr = hist.data_ptr<int32_t>();

            cudaLaunchKernelEx(&config, kernel_wrapper, topk_weights_ptr,
                               topk_indices_ptr, max_n_tiles, num_tokens,
                               gate_scale_ptr, topk_index_ptr, gate_index_ptr,
                               token_offs_pad_ptr, block_pid_map_ptr,
                               expt_offs_ptr, hist_ptr);

            break;
        }
            TORCH_CHECK(false, "Unsupported num experts: ", num_experts);
    }
}

void fused_routing(torch::Tensor& gating_output, torch::Tensor& topk_weights,
                   torch::Tensor& topk_indices, int64_t max_n_tiles,
                   int64_t topk, torch::Tensor& gate_scale,
                   torch::Tensor& topk_index, torch::Tensor& gate_index,
                   torch::Tensor& token_offs_pad, torch::Tensor& block_pid_map,
                   torch::Tensor& expt_offs, torch::Tensor& hist) {
    /*
    dispatch dtype
    */
    if (topk_indices.scalar_type() == at::ScalarType::Short &&
        gate_scale.scalar_type() == at::ScalarType::BFloat16) {
        routing_kernel_helper<__nv_bfloat16>(
            gating_output, topk_weights, topk_indices, max_n_tiles, topk,
            gate_scale, topk_index, gate_index, token_offs_pad, block_pid_map,
            expt_offs, hist);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", topk_indices.scalar_type(),
                    gate_scale.scalar_type());
    }
}