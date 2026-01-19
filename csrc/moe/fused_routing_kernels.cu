#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"
#include "../cub_helpers.h"
#include <cooperative_groups.h>

#define KERNEL_DEBUG 0
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

#define FUSED_ROUTING_CEIL_DIV(x, y) (x + ((y)-1)) / (y)

template <int NUM_EXPERTS>
__forceinline__ __device__ void collect_hist(
    typename cooperative_groups::cluster_group& handle,
    int32_t* __restrict__ hist, int32_t* __restrict__ global_hist) {
    auto cta_rank = handle.block_rank();
    auto cluster_size = handle.num_blocks();
    auto tid = threadIdx.x;
    int32_t __shared__ hist_buffer[NUM_EXPERTS];
    if (tid < NUM_EXPERTS) {
        hist_buffer[tid] = hist[tid];  // save
    }
    for (int stride = 1; stride <= cluster_size / 2; stride *= 2) {
        if (cta_rank % (stride * 2) == 0) {
            auto partner_cta = cta_rank + stride;
            if (partner_cta < cluster_size) {
                int32_t* partner_hist =
                    handle.map_shared_rank(hist, partner_cta);
                if (tid < NUM_EXPERTS) {
                    hist[tid] += partner_hist[tid];
                }
            }
        }
        handle.sync();
    }
    if (cta_rank == 0 && tid < NUM_EXPERTS) global_hist[tid] = hist[tid];
    if (tid < NUM_EXPERTS) hist[tid] = hist_buffer[tid];  // restore
}

template <int NUM_EXPERTS>
__forceinline__ __device__ void prefix_hist_CTA(
    cooperative_groups::cluster_group& handle, int32_t* __restrict__ hist,
    int32_t* __restrict__ prefix) {
    auto cta_rank = handle.block_rank();
    auto cluster_size = handle.num_blocks();
    auto tid = threadIdx.x;
    if (cta_rank > 0 && tid < NUM_EXPERTS) {
        int32_t* prev_hist = handle.map_shared_rank(hist, cta_rank - 1);
        prefix[tid] = prev_hist[tid];
    }

    handle.sync();  // ensure prefix has been initialized.
    if (cta_rank == 0) {
        for (auto rank = 2; rank < cluster_size; rank++) {
            auto prev_prefix = handle.map_shared_rank(prefix, rank - 1);
            auto dst_prefix = handle.map_shared_rank(prefix, rank);
            if (tid < NUM_EXPERTS) dst_prefix[tid] += prev_prefix[tid];
        }
    }
}

template <int NUM_EXPERTS, int topk, int NUM_BLOCK_SIZES, typename InValDtype,
          typename OutValDtype>
__global__ void fused_routing_kernel(
    InValDtype* __restrict__ topk_weights, int32_t* __restrict__ topk_indices,
    const int64_t max_n_tiles, const int64_t NUM_TOKENS,
    OutValDtype* __restrict__ gate_scale, int32_t* __restrict__ topk_index,
    int32_t* __restrict__ gate_index, int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr) {
    TORCH_CHECK(false, "unimplemented kernel");
}

template <>
__global__ void fused_routing_kernel<128, 4, 4, __nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices, const int64_t max_n_tiles,
    const int64_t NUM_TOKENS, __nv_bfloat16* __restrict__ gate_scale,
    int32_t* __restrict__ topk_index, int32_t* __restrict__ gate_index,
    int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr) {
    using InValDtype = __nv_bfloat16;
    using OutValDtype = __nv_bfloat16;
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int local_tid = threadIdx.x;
    int blockdimx = blockDim.x;
    int tid = local_tid + blockDim.x * blockIdx.x;
    int CTA_ID = blockIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    static constexpr int NUM_EXPERTS = 128;
    static constexpr int topk = 4;
    static constexpr int topk_padded = topk + 1;
    static constexpr int NUM_BLOCK_SIZES = 4;
    int ROWS_PER_THREADS = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, num_threads);
    int ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, gridDim.x);
    int HYPO_ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, 8);
    static constexpr int warp_size = 32;

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
    using BlockScan = cub::BlockScan<int, 512>;
    __shared__ typename BlockScan::TempStorage temp_storage_hist;
    __shared__
        typename BlockScan::TempStorage temp_storage_tiles[NUM_BLOCK_SIZES];
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
    __syncthreads();  // Ensure all threads complete zeroing before setting -1

#pragma unroll
    for (int i = local_tid; i < NUM_BLOCK_SIZES * (max_n_tiles);
         i += blockDim.x) {
        sm_hist[block_pid_offset + i] = -1;
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

        int expt0, expt1, expt2, expt3;
        int4 expts;
        if (i >= 0 && i < NUM_TOKENS) {
            expts = *reinterpret_cast<int4*>(topk_indices + i * topk);
            expt0 = expts.x;
            expt1 = expts.y;
            expt2 = expts.z;
            expt3 = expts.w;
            // expt0 = static_cast<int32_t>(topk_indices[i * topk + 0]);
            // expt1 = static_cast<int32_t>(topk_indices[i * topk + 1]);
            // expt2 = static_cast<int32_t>(topk_indices[i * topk + 2]);
            // expt3 = static_cast<int32_t>(topk_indices[i * topk + 3]);
        }
        int local_i = i - row;
        if (expt0 >= 0 && expt0 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded] =
                atomicAdd(local_hist + expt0, 1);
        if (expt1 >= 0 && expt1 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded + 1] =
                atomicAdd(local_hist + expt1, 1);
        if (expt2 >= 0 && expt2 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded + 2] =
                atomicAdd(local_hist + expt2, 1);
        if (expt3 >= 0 && expt3 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded + 3] =
                atomicAdd(local_hist + expt3, 1);
    }
    cluster.sync();
    int32_t* global_hist =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
    collect_hist<NUM_EXPERTS>(cluster, local_hist, global_hist);
    /* phase 2*/
    // compute expert_across_prefixsum, dst_experts[i] =
    // sum_{p=0}^{j-1}{local_hist[p]}, where i is expert_id, j is CTA_ID
    // TODO(ijpq): we can leverage warpscan to improve this process, but need
    // assign warp threads into [NUM_CTAS, NUM_EXPERTS] carefully.
    prefix_hist_CTA<NUM_EXPERTS>(cluster, sm_hist + local_hist_offset,
                                 sm_hist + expert_across_offset);
    cluster.sync();
    int h = 0;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (CTA_ID == 0 && local_tid < NUM_EXPERTS) {
        int32_t* global_hist_sm0 =
            reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
        h = global_hist_sm0[local_tid];
    }
    if (CTA_ID == 0) {
        // compute global hist prefixsum
        int block_exclusive_res = 0;
        int block_reduce = 0;
        BlockScan(temp_storage_hist)
            .ExclusiveSum(h, block_exclusive_res, block_reduce);
        int32_t* hist_sum = reinterpret_cast<int32_t*>(
            sm_hist + global_hist_exclusivesum_offset);
        if (local_tid < NUM_EXPERTS) {
            hist_sum[local_tid] = block_exclusive_res;
            if (local_tid == 0) hist_sum[NUM_EXPERTS] = block_reduce;
        }
#pragma unroll
        for (int size = 0; size < NUM_BLOCK_SIZES; size += 1) {
            int BLOCK_M_LOG2_START = 4;
            int block_m_log2 = BLOCK_M_LOG2_START + size;
            int block_m = 1 << block_m_log2;  // block_m = 16, 32,64,128
            int n_tiles = FUSED_ROUTING_CEIL_DIV(h, block_m);
            int tiles_exclusive_res = 0;
            int tiles_reduce = 0;
            BlockScan(temp_storage_tiles[size])
                .ExclusiveSum(n_tiles, tiles_exclusive_res, tiles_reduce);
            int32_t* pid_map_row =
                reinterpret_cast<int32_t*>(sm_hist + block_pid_offset) +
                size * max_n_tiles;
            int32_t* token_offs_pad =
                reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
            if (local_tid < NUM_EXPERTS) {
                token_offs_pad[size * (NUM_EXPERTS + 1) + local_tid] =
                    tiles_exclusive_res;
                if (local_tid == 0)
                    token_offs_pad[size * (NUM_EXPERTS + 1) + NUM_EXPERTS] =
                        tiles_reduce;
                int tile_start = tiles_exclusive_res;
                for (int block_idx = 0; block_idx < n_tiles; block_idx++) {
                    int packed_val = (block_idx << 16) | local_tid;
                    pid_map_row[(tile_start + block_idx)] = packed_val;
                }
            }
        }
    }
    cluster.sync();

    // WB global memory
    int token_offs_pad_size = NUM_BLOCK_SIZES * (NUM_EXPERTS + 1);
    int pid_map_size = NUM_BLOCK_SIZES * (max_n_tiles);

    if (CTA_ID == 0) {
        int32_t* token_offs_pad =
            reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
        int32_t* block_pid =
            reinterpret_cast<int32_t*>(sm_hist + block_pid_offset);
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
        hist_ptr[local_tid] = global_hist_sm[local_tid];
        if (local_tid == 0) expt_offs_ptr[NUM_EXPERTS] = hist_sum[NUM_EXPERTS];
    }
    cluster.sync();

    /* phase 3*/

    int32_t* prior_contrib =
        reinterpret_cast<int32_t*>(sm_hist + expert_across_offset);

    /*
    WE HAVE TO SYNC TO CTA'S LOCAL SM SINCE MAP_SHARED_RANK LEADS TO
    cudaErrorLaunchFailure.
    */
    int32_t* hist_sum_sm0 = cluster.map_shared_rank(
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset),
        0);
    int32_t* hist_sum_local =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset);
    if (local_tid < NUM_EXPERTS)
        hist_sum_local[local_tid] = hist_sum_sm0[local_tid];
    cluster.sync();

#pragma unroll
    for (int i = row + local_tid; i < row_end; i += blockDim.x) {
        int local_i = i - row;
        int topk_idx_stride = topk, topk_val_stride = topk;
        for (int k = 0; k < topk; k++) {
            if (i >= 0 && i < NUM_TOKENS) {
                int expert_id = topk_indices[i * topk_idx_stride + k];
                if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
                    InValDtype val = topk_weights[i * topk_val_stride + k];
                    int flat_idx = i * topk + k;
                    int expert_base = hist_sum_local[expert_id];
                    int expert_prior = prior_contrib[expert_id];
                    int expert_local =
                        local_offset_sm[local_i * topk_padded + k];
                    int global_pos = expert_base + expert_prior + expert_local;

                    if (global_pos < NUM_TOKENS * topk &&
                        flat_idx < NUM_TOKENS * topk) {
                        gate_scale[global_pos] = static_cast<OutValDtype>(val);
                        topk_index[global_pos] = flat_idx;
                        gate_index[flat_idx] = global_pos;
                    }
                }
            }
        }
    }
}

template <>
__global__ void fused_routing_kernel<32, 4, 4, __nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices, const int64_t max_n_tiles,
    const int64_t NUM_TOKENS, __nv_bfloat16* __restrict__ gate_scale,
    int32_t* __restrict__ topk_index, int32_t* __restrict__ gate_index,
    int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr) {
    using InValDtype = __nv_bfloat16;
    using OutValDtype = __nv_bfloat16;
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int local_tid = threadIdx.x;
    int tid = local_tid + blockDim.x * blockIdx.x;
    int CTA_ID = blockIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    static constexpr int NUM_EXPERTS = 32;
    static constexpr int topk = 4;
    static constexpr int topk_padded = topk + 1;
    static constexpr int NUM_BLOCK_SIZES = 4;
    int ROWS_PER_THREADS = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, num_threads);
    int ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, gridDim.x);
    int HYPO_ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, 8);
    static constexpr int warp_size = 32;

    // shared memory layout
    /*
    Assumingly, shared Mem is organized as this layout:
    [NUM_EXPERTS] : global hist !only 0
    [NUM_EXPERTS] : local histgram for each threadblock
    [NUM_EXPERTS+1]: global hist exclusive-sum !only 0
    [NUM_BLOCK_SIZES, NUM_EXPERTS+1] : token_offs_pad
    [NUM_BLOCK_SIZES, max_n_tiles] : block_pid
    [NUM_EXPERTS]: exclusivesum for experts
    [ROWS_PER_CTA*topkpadded]: local_offset
    */
    using WarpScan = cub::WarpScan<int>;
    __shared__ typename WarpScan::TempStorage
        temp_storage_hist[NUM_EXPERTS / warp_size];
    __shared__
        typename WarpScan::TempStorage temp_storage_tiles[NUM_BLOCK_SIZES];
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
    int shared_mem_size = local_offset_offset + topk_padded * ROWS_PER_CTA;

#pragma unroll
    for (int i = local_tid; i < shared_mem_size; i += blockDim.x) {
        sm_hist[i] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = local_tid; i < NUM_BLOCK_SIZES * (max_n_tiles);
         i += blockDim.x) {
        sm_hist[block_pid_offset + i] = -1;
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
        int expt0, expt1, expt2, expt3;
        int4 expts;
        if (i >= 0 && i < NUM_TOKENS) {
            expts = *reinterpret_cast<int4*>(topk_indices + i * topk);
            expt0 = expts.x;
            expt1 = expts.y;
            expt2 = expts.z;
            expt3 = expts.w;
            // expt0 = static_cast<int32_t>(topk_indices[i * topk + 0]);
            // expt1 = static_cast<int32_t>(topk_indices[i * topk + 1]);
            // expt2 = static_cast<int32_t>(topk_indices[i * topk + 2]);
            // expt3 = static_cast<int32_t>(topk_indices[i * topk + 3]);
        }
        int local_i = i - row;
        if (expt0 >= 0 && expt0 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded] =
                atomicAdd(local_hist + expt0, 1);
        if (expt1 >= 0 && expt1 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded + 1] =
                atomicAdd(local_hist + expt1, 1);
        if (expt2 >= 0 && expt2 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded + 2] =
                atomicAdd(local_hist + expt2, 1);
        if (expt3 >= 0 && expt3 < NUM_EXPERTS)
            local_offset_sm[local_i * topk_padded + 3] =
                atomicAdd(local_hist + expt3, 1);
    }
    cluster.sync();
    int32_t* global_hist =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
    collect_hist<NUM_EXPERTS>(cluster, local_hist, global_hist);

    /* phase 2*/
    int warp_id = threadIdx.x / 32;
    // compute expert_across_prefixsum, dst_experts[i] =
    // sum_{p=0}^{j-1}{local_hist[p]}, where i is expert_id, j is CTA_ID
    // TODO(ijpq): we can leverage warpscan to improve this process, but need
    // assign warp threads into [NUM_CTAS, NUM_EXPERTS] carefully.
    prefix_hist_CTA<NUM_EXPERTS>(cluster, sm_hist + local_hist_offset,
                                 sm_hist + expert_across_offset);
    cluster.sync();
    if (CTA_ID == 0 && warp_id < NUM_BLOCK_SIZES) {
        int32_t* global_hist_sm0 =
            reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
        int lane_id = threadIdx.x % 32;
        int h = global_hist_sm0[lane_id];
        // compute global hist prefixsum
        if (warp_id == 0) {
            int exclusive_res = 0;
            int warp_reduce = 0;
            WarpScan(temp_storage_hist[0])
                .ExclusiveSum(h, exclusive_res, warp_reduce);
            int32_t* hist_sum = reinterpret_cast<int32_t*>(
                sm_hist + global_hist_exclusivesum_offset);
            hist_sum[local_tid] = exclusive_res;
            if (local_tid == 0) hist_sum[NUM_EXPERTS] = warp_reduce;
        }

        // align the data with triton's matmul_ogs
        int BLOCK_M_LOG2_START = 4;
        int block_m_log2 = BLOCK_M_LOG2_START + warp_id;
        int block_m = 1 << block_m_log2;  // block_m = 16, 32,64,128
        int32_t* pid_map_row =
            reinterpret_cast<int32_t*>(sm_hist + block_pid_offset) +
            warp_id * max_n_tiles;
        int n_tiles = (h + block_m - 1) / block_m;
        int warp_reduce = 0;
        int exclusive_res = 0;
        WarpScan(temp_storage_tiles[warp_id])
            .ExclusiveSum(n_tiles, exclusive_res, warp_reduce);

        int32_t* token_offs_pad =
            reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
        token_offs_pad[warp_id * (NUM_EXPERTS + 1) + lane_id] = exclusive_res;
        if (lane_id == 0)
            token_offs_pad[warp_id * (NUM_EXPERTS + 1) + NUM_EXPERTS] =
                warp_reduce;

        int tile_start = exclusive_res;
        for (int block_idx = 0; block_idx < n_tiles; block_idx++) {
            int packed_val = (block_idx << 16) | lane_id;
            pid_map_row[(tile_start + block_idx)] = packed_val;
        }
    }
    cluster.sync();

    // WB global memory
    int token_offs_pad_size = NUM_BLOCK_SIZES * (NUM_EXPERTS + 1);
    int pid_map_size = NUM_BLOCK_SIZES * (max_n_tiles);

    if (CTA_ID == 0) {
        int32_t* token_offs_pad =
            reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
        int32_t* block_pid =
            reinterpret_cast<int32_t*>(sm_hist + block_pid_offset);
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
        hist_ptr[local_tid] = global_hist_sm[local_tid];
        if (local_tid == 0) expt_offs_ptr[NUM_EXPERTS] = hist_sum[NUM_EXPERTS];
    }
    cluster.sync();

    /* phase 3*/

    int32_t* prior_contrib =
        reinterpret_cast<int32_t*>(sm_hist + expert_across_offset);

    /*
    WE HAVE TO SYNC TO CTA'S LOCAL SM SINCE MAP_SHARED_RANK LEADS TO
    cudaErrorLaunchFailure.
    */
    int32_t* hist_sum_sm0 = cluster.map_shared_rank(
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset),
        0);
    int32_t* hist_sum_local =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset);
    if (local_tid < NUM_EXPERTS)
        hist_sum_local[local_tid] = hist_sum_sm0[local_tid];
    cluster.sync();

#pragma unroll
    for (int i = row + local_tid; i < row_end; i += blockDim.x) {
        int local_i = i - row;
        int topk_idx_stride = topk, topk_val_stride = topk;
        for (int k = 0; k < topk; k++) {
            if (i >= 0 && i < NUM_TOKENS) {
                int expert_id = topk_indices[i * topk_idx_stride + k];
                if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
                    InValDtype val = topk_weights[i * topk_val_stride + k];
                    int flat_idx = i * topk + k;
                    int expert_base = hist_sum_local[expert_id];
                    int expert_prior = prior_contrib[expert_id];
                    int expert_local =
                        local_offset_sm[local_i * topk_padded + k];
                    int global_pos = expert_base + expert_prior + expert_local;

                    if (global_pos < NUM_TOKENS * topk &&
                        flat_idx < NUM_TOKENS * topk) {
                        gate_scale[global_pos] = static_cast<OutValDtype>(val);
                        topk_index[global_pos] = flat_idx;
                        gate_index[flat_idx] = global_pos;
                    }
                }
            }
        }
    }
}

}  // namespace moe
}  // namespace vllm

template <typename IdxType, typename InValType, typename OutValType>
void routing_kernel_helper(torch::Tensor& gating_output,
                           torch::Tensor& topk_weights,
                           torch::Tensor& topk_indices, int64_t max_n_tiles,
                           int64_t topk, torch::Tensor& gate_scale,
                           torch::Tensor& topk_index, torch::Tensor& gate_index,
                           torch::Tensor& token_offs_pad,
                           torch::Tensor& block_pid_map,
                           torch::Tensor& expt_offs, torch::Tensor& hist) {
    /*
    dispatch config
    */
    TORCH_CHECK(topk == 4, "");
    constexpr int const_topk = 4;
    constexpr int const_topk_padded = const_topk + 1;
    constexpr int NUM_BLOCK_SIZES = 4;
    const auto num_experts = gating_output.size(-1);
    const auto num_tokens = gating_output.numel() / num_experts;
#if KERNEL_DEBUG
    std::cout << "=== fused_routing called ===" << std::endl;
    std::cout << "  num_tokens: " << num_tokens << std::endl;
    std::cout << "  num_experts: " << num_experts << std::endl;
    std::cout << "  max_n_tiles: " << max_n_tiles << std::endl;
    std::cout << "  topk: " << topk << std::endl;
#endif

    switch (num_experts) {
        case 32: {
            auto warp_size = 32;
            int cluster_size = 0;
            int THREAD_PER_CTA = 512;

            // hypothetical config
            int hypo_cluster_size = 8;
            size_t rows_per_cta =
                (num_tokens + hypo_cluster_size - 1) / hypo_cluster_size;
            size_t global_hist_size = num_experts;
            size_t local_hist_size = num_experts;
            size_t global_hist_prefix_size = num_experts + 1;
            size_t token_offs_pad_size = NUM_BLOCK_SIZES * (num_experts + 1);
            size_t block_pid_size = NUM_BLOCK_SIZES * (max_n_tiles);
            size_t prefix_experts_size = num_experts;
            size_t local_offset_size = const_topk_padded * rows_per_cta;
            auto kernel_wrapper = &(
                vllm::moe::fused_routing_kernel<32, const_topk, NUM_BLOCK_SIZES,
                                                InValType, OutValType>);

            cudaFuncAttributes attr;
            cudaLaunchConfig_t config = {};
            auto cuda_error = cudaFuncGetAttributes(&attr, kernel_wrapper);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error));
            size_t static_smem_size = attr.sharedSizeBytes;
            size_t required_dynamicSmemBytes =
                (global_hist_size + local_hist_size + global_hist_prefix_size +
                 token_offs_pad_size + block_pid_size + prefix_experts_size +
                 local_offset_size) *
                sizeof(int32_t);
            size_t requried_sm_size =
                static_smem_size + required_dynamicSmemBytes;
            config.dynamicSmemBytes = required_dynamicSmemBytes + 1024;

            // dev id
            int dev_id = 0;
            cuda_error = cudaGetDevice(&dev_id);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error));

            // max sm size
            int max_hw_limit = 0;
            cuda_error = cudaDeviceGetAttribute(
                &max_hw_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);
            TORCH_CHECK(requried_sm_size <= max_hw_limit,
                        cudaGetErrorString(cuda_error));

            cuda_error = cudaFuncSetAttribute(
                kernel_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize,
                config.dynamicSmemBytes);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))

            cuda_error = cudaFuncSetAttribute(
                kernel_wrapper, cudaFuncAttributeNonPortableClusterSizeAllowed,
                1);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))

            cuda_error = cudaFuncGetAttributes(&attr, kernel_wrapper);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
            cuda_error = cudaOccupancyMaxPotentialClusterSize(
                &cluster_size, kernel_wrapper, &config);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
#if KERNEL_DEBUG
            std::cout << "  binaryVersion: " << attr.binaryVersion << std::endl;
            std::cout << "  maxDynamicSharedSizeBytes: "
                      << attr.maxDynamicSharedSizeBytes << std::endl;
            std::cout << "  sharedSizeBytes: " << attr.sharedSizeBytes
                      << std::endl;
            std::cout << "cluster size: " << cluster_size << std::endl;
            std::cout << "Req Smem: " << config.dynamicSmemBytes / 1024.f
                      << " Kbytes" << std::endl;
            if (cluster_size < hypo_cluster_size) {
                std::cout << "cluster size error" << cluster_size << std::endl;
            }
#endif
            auto grid_dim = dim3(hypo_cluster_size, 1, 1);
            // recompute config
            if (cluster_size > hypo_cluster_size) {
                size_t rows_per_cta =
                    (num_tokens + cluster_size - 1) / cluster_size;
                local_offset_size = const_topk_padded * rows_per_cta;
                required_dynamicSmemBytes =
                    (global_hist_size + local_hist_size +
                     global_hist_prefix_size + token_offs_pad_size +
                     block_pid_size + prefix_experts_size + local_offset_size) *
                    sizeof(int32_t);

                requried_sm_size = static_smem_size + required_dynamicSmemBytes;
                config.dynamicSmemBytes = required_dynamicSmemBytes + 1024;
                cuda_error = cudaFuncSetAttribute(
                    kernel_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    config.dynamicSmemBytes);
                TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
                grid_dim = dim3(cluster_size, 1, 1);
            }
            config.blockDim = dim3(THREAD_PER_CTA, 1, 1);
            config.gridDim = grid_dim;

            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = grid_dim.x;
            attribute[0].val.clusterDim.y = 1;
            attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute;
            config.numAttrs = 1;
            const cudaStream_t current_stream =
                at::cuda::getCurrentCUDAStream();
            config.stream = current_stream;

            auto topk_weights_ptr =
                reinterpret_cast<InValType*>(topk_weights.data_ptr());
            auto gate_scale_ptr =
                reinterpret_cast<OutValType*>(gate_scale.data_ptr());
            auto topk_indices_ptr = topk_indices.data_ptr<IdxType>();
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
        case 128: {
            auto warp_size = 32;
            int cluster_size = 0;
            int THREAD_PER_CTA = 512;

            int hypo_cluster_size = 8;
            size_t rows_per_cta =
                (num_tokens + hypo_cluster_size - 1) / hypo_cluster_size;
            size_t global_hist_size = num_experts;
            size_t local_hist_size = num_experts;
            size_t global_hist_prefix_size = num_experts + 1;
            size_t token_offs_pad_size = NUM_BLOCK_SIZES * (num_experts + 1);
            size_t block_pid_size = NUM_BLOCK_SIZES * (max_n_tiles);
            size_t prefix_experts_size = num_experts;
            size_t local_offset_size = const_topk_padded * rows_per_cta;
            auto kernel_wrapper =
                &(vllm::moe::fused_routing_kernel<
                    128, const_topk, NUM_BLOCK_SIZES, InValType, OutValType>);

            cudaFuncAttributes attr;
            cudaLaunchConfig_t config = {};
            auto cuda_error = cudaFuncGetAttributes(&attr, kernel_wrapper);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error));
            size_t static_smem_size = attr.sharedSizeBytes;
            size_t required_dynamicSmemBytes =
                (global_hist_size + local_hist_size + global_hist_prefix_size +
                 token_offs_pad_size + block_pid_size + prefix_experts_size +
                 local_offset_size) *
                sizeof(int32_t);
            size_t requried_sm_size =
                static_smem_size + required_dynamicSmemBytes;
            config.dynamicSmemBytes = required_dynamicSmemBytes + 1024;

            // dev id
            int dev_id = 0;
            cuda_error = cudaGetDevice(&dev_id);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error));

            // max sm size
            int max_hw_limit = 0;
            cuda_error = cudaDeviceGetAttribute(
                &max_hw_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);
            TORCH_CHECK(requried_sm_size <= max_hw_limit,
                        cudaGetErrorString(cuda_error));

            cuda_error = cudaFuncSetAttribute(
                kernel_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize,
                config.dynamicSmemBytes);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))

            cuda_error = cudaFuncSetAttribute(
                kernel_wrapper, cudaFuncAttributeNonPortableClusterSizeAllowed,
                1);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))

            cuda_error = cudaFuncGetAttributes(&attr, kernel_wrapper);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
            cuda_error = cudaOccupancyMaxPotentialClusterSize(
                &cluster_size, kernel_wrapper, &config);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
#if KERNEL_DEBUG
            std::cout << "  binaryVersion: " << attr.binaryVersion << std::endl;
            std::cout << "  maxDynamicSharedSizeBytes: "
                      << attr.maxDynamicSharedSizeBytes << std::endl;
            std::cout << "  sharedSizeBytes: " << attr.sharedSizeBytes
                      << std::endl;
            std::cout << "cluster size: " << cluster_size << std::endl;
            std::cout << "Req Smem: " << config.dynamicSmemBytes / 1024.f
                      << " Kbytes" << std::endl;
            if (cluster_size < hypo_cluster_size) {
                std::cout << "cluster size error" << cluster_size << std::endl;
            }
#endif
            auto grid_dim = dim3(hypo_cluster_size, 1, 1);
            // recompute config
            if (cluster_size > hypo_cluster_size) {
                size_t rows_per_cta =
                    (num_tokens + cluster_size - 1) / cluster_size;
                local_offset_size = const_topk_padded * rows_per_cta;
                required_dynamicSmemBytes =
                    (global_hist_size + local_hist_size +
                     global_hist_prefix_size + token_offs_pad_size +
                     block_pid_size + prefix_experts_size + local_offset_size) *
                    sizeof(int32_t);

                requried_sm_size = static_smem_size + required_dynamicSmemBytes;
                config.dynamicSmemBytes = required_dynamicSmemBytes + 1024;
                cuda_error = cudaFuncSetAttribute(
                    kernel_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    config.dynamicSmemBytes);
                TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
                grid_dim = dim3(cluster_size, 1, 1);
            }

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
            const cudaStream_t current_stream =
                at::cuda::getCurrentCUDAStream();
            config.stream = current_stream;

            auto topk_weights_ptr =
                reinterpret_cast<InValType*>(topk_weights.data_ptr());
            auto gate_scale_ptr =
                reinterpret_cast<OutValType*>(gate_scale.data_ptr());
            auto topk_indices_ptr = topk_indices.data_ptr<IdxType>();
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
    if (topk_indices.scalar_type() == at::ScalarType::Int &&
        gate_scale.scalar_type() == at::ScalarType::BFloat16 &&
        topk_weights.scalar_type() == at::ScalarType::BFloat16) {
        int int4_alignment = 16;
        assert(reinterpret_cast<uintptr_t>(topk_indices.data_ptr()) %
                   int4_alignment ==
               0);
        routing_kernel_helper<int32_t, __nv_bfloat16, __nv_bfloat16>(
            gating_output, topk_weights, topk_indices, max_n_tiles, topk,
            gate_scale, topk_index, gate_index, token_offs_pad, block_pid_map,
            expt_offs, hist);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", topk_indices.scalar_type(),
                    gate_scale.scalar_type());
    }
}