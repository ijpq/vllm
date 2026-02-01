#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"
#include "../cub_helpers.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

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

namespace fused_routing {

#if KERNEL_DEBUG
__device__ void debug_cp_async_int32(const int32_t* sm_base,
                                     int offset_in_int32, int copy_size_bytes,
                                     int total_smem_int32, const char* label,
                                     int tid = 0) {
    int byte_offset = offset_in_int32 * sizeof(int32_t);
    int total_bytes = total_smem_int32 * sizeof(int32_t);

    int write_start = byte_offset;
    int write_end = byte_offset + copy_size_bytes;

    bool in_bounds = (write_start >= 0) && (write_end <= total_bytes);

    const void* actual_ptr = sm_base + offset_in_int32;
    bool aligned = ((reinterpret_cast<uintptr_t>(actual_ptr) & 0x7) == 0);

    if (threadIdx.x == tid) {
        printf(
            "[%s] offset=%d (int32) = %d bytes\n       write_range=[%d, %d) "
            "bytes, total_smem=%d bytes\n       in_bounds=%d, aligned=%d\n     "
            "  actual_ptr=%p\n       tid=%d\n",
            label, offset_in_int32, byte_offset, write_start, write_end,
            total_bytes, in_bounds, aligned, actual_ptr, tid);
    }
}
#endif

#define FUSED_ROUTING_CEIL_DIV(x, y) (((x) + (y)-1) / (y))

#define MAKE_ALIGNMENT_DIFF(x, y) ((((x) + (y)-1) / (y) * (y)) - (x))

__device__ int layout_addr(int row, int col) {
    /*
     * Swizzled address layout for local_offset_sm to avoid shared memory bank
     * conflicts. Memory Layout: ┌─────────────────────── warp 0 (128 elements)
     * ───────────────────────┐ │  col=0 (32 elem)  │  col=1 (32 elem)  │  col=2
     * (32 elem)  │  col=3  │ │  addr: 0-31       │  addr: 32-63      │  addr:
     * 64-95      │  96-127 │ │  row 0-31         │  row 0-31         │  row
     * 0-31         │ row 0-31│
     *   └──────────────────────────────────────────────────────────────────────┘
     *   ┌─────────────────────── warp 1 (128 elements) ───────────────────────┐
     *   │  col=0 (32 elem)  │  col=1 (32 elem)  │  col=2 (32 elem)  │  col=3  │
     *   │  addr: 128-159    │  addr: 160-191    │  addr: 192-223    │ 224-255 │
     *   │  row 32-63        │  row 32-63        │  row 32-63        │row 32-63│
     *   └──────────────────────────────────────────────────────────────────────┘
     *   ...
     *   - warp_offset:  base address for each warp block (128 per warp)
     *   - group_offset: offset within warp for each column (32 per column)
     *   - elem_idx:     lane position within the column group
     *
     * Bank Conflict Analysis (col=0 access within one warp):
     *   thread 0  -> addr = 0,  bank = 0
     *   thread 1  -> addr = 1,  bank = 1
     *   ...
     *   thread 31 -> addr = 31, bank = 31
     *   All 32 threads access different banks -> NO conflict
     *
     * Shared Memory Size:
     *   num_warps = CEIL_DIV(rows_per_cta, 32)
     *   local_offset_size = num_warps * 32 * topk
     */
    constexpr int COLS = 4;  // topk
    constexpr int warp_size = 32;
    int elem_idx = row % warp_size;
    int group = col;
    int group_offset = col * warp_size;
    int warp_offset = row / warp_size * warp_size * COLS;
    int addr = warp_offset + group_offset + elem_idx;
    return addr;
}

__device__ __forceinline__ void cp_async_cg_pred(void* smem_ptr,
                                                 const void* glob_ptr,
                                                 bool pred = true) {
    volatile uint32_t smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], 16;\n"
        "}\n" ::"r"((int)pred),
        "r"(smem), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_ca_pred(void* smem_ptr,
                                                 const void* glob_ptr,
                                                 bool pred = true) {
    volatile uint32_t smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], 8;\n"
        "}\n" ::"r"((int)pred),
        "r"(smem), "l"(glob_ptr));
}

__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

template <int NUM_EXPERTS>
__forceinline__ __device__ void _collect_hist(
    typename cooperative_groups::grid_group& handle, int32_t* __restrict__ hist,
    int32_t* __restrict__ hist_ptr, int32_t* __restrict__ global_hist) {
    auto tid = threadIdx.x;
    // All CTAs atomic add their local histogram to global hist_ptr
    if (tid < NUM_EXPERTS) atomicAdd(&hist_ptr[tid], hist[tid]);

    // Sync to ensure all CTAs have contributed
    handle.sync();

    // CTA 0 copies the aggregated global histogram to its shared memory
    int num_steps = NUM_EXPERTS / 4;
    if (blockIdx.x == 0) {
        if (tid < num_steps)
            *reinterpret_cast<int4*>(global_hist + tid * 4) =
                *reinterpret_cast<int4*>(hist_ptr + tid * 4);
    }
}

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
__forceinline__ __device__ void _prefix_hist_CTA(
    typename cooperative_groups::grid_group& handle, int32_t* __restrict__ hist,
    int32_t* __restrict__ prefix) {
    // prefix[i, :] = sum_{j=0}^{i-1}{local_hist[j, :]} (exclusive prefix sum
    // per expert) Step 1: Each CTA writes its local hist to prefix buffer
    int tid = threadIdx.x;
    int CTA_ID = blockIdx.x;
    int num_steps = NUM_EXPERTS / 4;
    if (CTA_ID < handle.num_blocks() - 1 && tid < num_steps)
        *reinterpret_cast<int4*>(prefix + (CTA_ID + 1) * NUM_EXPERTS +
                                 tid * 4) =
            *reinterpret_cast<int4*>(hist + tid * 4);

    handle.sync();

    // Step 2: CTA 0 performs sequential exclusive prefix sum
    if (CTA_ID == 0) {
        // First block has no prior contribution
        if (tid < num_steps) *reinterpret_cast<int4*>(prefix + tid * 4) = make_int4(0,0,0,0);
        // if (tid < NUM_EXPERTS) {
        //     prefix[tid] = 0;
        // }

        // First, compute exclusive prefix sum in-place
        for (int bdx = 1; bdx < handle.num_blocks(); bdx++) {
            if (tid < num_steps) {
                int4* current_ptr = reinterpret_cast<int4*>(
                    prefix + bdx * NUM_EXPERTS + tid * 4);
                int4* prev_ptr = reinterpret_cast<int4*>(
                    prefix + (bdx - 1) * NUM_EXPERTS + tid * 4);
                int4 current_val = *current_ptr;
                int4 prev_val = *prev_ptr;
                current_val.x += prev_val.x;
                current_val.y += prev_val.y;
                current_val.z += prev_val.z;
                current_val.w += prev_val.w;
                *current_ptr = current_val;
            }
            // if (tid < NUM_EXPERTS) {
            //     prefix[bdx * NUM_EXPERTS + tid] +=
            //         prefix[(bdx - 1) * NUM_EXPERTS + tid];
            // }
            __syncthreads();
        }
    }
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
// ====================== TopK Softmax Device Functions =======================
// Fused topk + softmax computation for MoE routing
// Supports renormalize=True, topk=4 case

template <int NUM_EXPERTS, int THREADS_PER_ROW>
__device__ __forceinline__ void compute_topk_softmax_row(
    const __nv_bfloat16* __restrict__ router_logits_row,
    float* __restrict__ topk_weights_out,    // output: top-4 weights (floats)
    int32_t* __restrict__ topk_indices_out,  // output: top-4 expert indices
    int thread_group_idx,                    // position within thread group
    int lane_id,                             // lane within warp
    int warp_id                              // warp id within CTA
) {
    constexpr int topk = 4;
    constexpr int WARP_SIZE_LOCAL = 32;

    // Each thread group processes one row
    // THREADS_PER_ROW threads collaborate on one row of NUM_EXPERTS elements
    static constexpr int ELTS_PER_THREAD = NUM_EXPERTS / THREADS_PER_ROW;

    // Load router logits into registers and convert to float
    float row_chunk[ELTS_PER_THREAD];
    int first_elt = thread_group_idx * ELTS_PER_THREAD;

#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] = __bfloat162float(router_logits_row[first_elt + i]);
    }

    // Step 1: Find max for numerical stability
    float thread_max = row_chunk[0];
#pragma unroll
    for (int i = 1; i < ELTS_PER_THREAD; i++) {
        thread_max = fmaxf(thread_max, row_chunk[i]);
    }

// Reduce max across threads in the group
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max =
            fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    // Step 2: Compute exp(x - max) and sum
    float row_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] = expf(row_chunk[i] - thread_max);
        row_sum += row_chunk[i];
    }

// Reduce sum across threads in the group
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask);
    }

    // Step 3: Normalize to get softmax probabilities
    float reciprocal_sum = 1.0f / row_sum;
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] *= reciprocal_sum;
    }

    // Step 4: Find top-4 values and indices using iterative argmax
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_THREAD * THREADS_PER_ROW;

    float selected_sum = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < topk; k_idx++) {
        // Find local max
        float max_val = row_chunk[0];
        int expert = first_elt;

#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i++) {
            if (row_chunk[i] > max_val) {
                max_val = row_chunk[i];
                expert = first_elt + i;
            }
        }

// Reduce across thread group using butterfly pattern
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, max_val, mask);
            int other_expert = __shfl_xor_sync(0xffffffff, expert, mask);

            // Prefer lower indices for tie-breaking
            if (other_max > max_val ||
                (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Thread 0 in group writes result
        if (thread_group_idx == 0) {
            topk_weights_out[k_idx] = max_val;
            topk_indices_out[k_idx] = expert;
            selected_sum += max_val;
        }

        // Clear the winning value for next iteration
        if (k_idx + 1 < topk) {
            int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            int thread_to_clear = (expert / ELTS_PER_THREAD) % THREADS_PER_ROW;

            if (thread_group_idx == thread_to_clear) {
                int offset_for_expert = expert % ELTS_PER_THREAD;
                row_chunk[offset_for_expert] = -10000.0f;
            }
        }
    }

    // Step 5: Renormalize top-k weights to sum to 1.0
    if (thread_group_idx == 0) {
        float denom = (selected_sum > 0.0f) ? selected_sum : 1.0f;
#pragma unroll
        for (int k_idx = 0; k_idx < topk; k_idx++) {
            topk_weights_out[k_idx] /= denom;
        }
    }
}

// Specialized version for 128 experts (4 threads per row, 32 elements per
// thread)
template <>
__device__ __forceinline__ void compute_topk_softmax_row<128, 4>(
    const __nv_bfloat16* __restrict__ router_logits_row,
    float* __restrict__ topk_weights_out,
    int32_t* __restrict__ topk_indices_out, int thread_group_idx, int lane_id,
    int warp_id) {
    constexpr int NUM_EXPERTS = 128;
    constexpr int THREADS_PER_ROW = 4;
    constexpr int topk = 4;
    constexpr int ELTS_PER_THREAD = NUM_EXPERTS / THREADS_PER_ROW;  // 32

    // Load router logits into registers and convert to float
    float row_chunk[ELTS_PER_THREAD];
    int first_elt = thread_group_idx * ELTS_PER_THREAD;

// Vectorized load (8 bfloat16 at a time = 16 bytes)
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i += 2) {
        __nv_bfloat162 val = *reinterpret_cast<const __nv_bfloat162*>(
            &router_logits_row[first_elt + i]);
        float2 f2 = __bfloat1622float2(val);
        row_chunk[i] = f2.x;
        row_chunk[i + 1] = f2.y;
    }

    // Step 1: Find max for numerical stability
    float thread_max = row_chunk[0];
#pragma unroll
    for (int i = 1; i < ELTS_PER_THREAD; i++) {
        thread_max = fmaxf(thread_max, row_chunk[i]);
    }

// Reduce max across 4 threads
#pragma unroll
    for (int mask = 2; mask > 0; mask /= 2) {
        thread_max =
            fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    // Step 2: Compute exp(x - max) and sum
    float row_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] = expf(row_chunk[i] - thread_max);
        row_sum += row_chunk[i];
    }

// Reduce sum across 4 threads
#pragma unroll
    for (int mask = 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask);
    }

    // Step 3: Normalize to get softmax probabilities
    float reciprocal_sum = 1.0f / row_sum;
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] *= reciprocal_sum;
    }

    // Step 4: Find top-4 values and indices using iterative argmax
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_THREAD * THREADS_PER_ROW;

    float selected_sum = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < topk; k_idx++) {
        // Find local max
        float max_val = row_chunk[0];
        int expert = first_elt;

#pragma unroll
        for (int i = 0; i < ELTS_PER_THREAD; i++) {
            if (row_chunk[i] > max_val) {
                max_val = row_chunk[i];
                expert = first_elt + i;
            }
        }

// Reduce across 4 threads using butterfly pattern
#pragma unroll
        for (int mask = 2; mask > 0; mask /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, max_val, mask);
            int other_expert = __shfl_xor_sync(0xffffffff, expert, mask);

            if (other_max > max_val ||
                (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Thread 0 in group writes result
        if (thread_group_idx == 0) {
            topk_weights_out[k_idx] = max_val;
            topk_indices_out[k_idx] = expert;
            selected_sum += max_val;
        }

        // Clear the winning value for next iteration
        if (k_idx + 1 < topk) {
            int thread_to_clear = (expert / ELTS_PER_THREAD) % THREADS_PER_ROW;

            if (thread_group_idx == thread_to_clear) {
                int offset_for_expert = expert % ELTS_PER_THREAD;
                row_chunk[offset_for_expert] = -10000.0f;
            }
        }
    }

    // Step 5: Renormalize top-k weights to sum to 1.0
    if (thread_group_idx == 0) {
        float denom = (selected_sum > 0.0f) ? selected_sum : 1.0f;
#pragma unroll
        for (int k_idx = 0; k_idx < topk; k_idx++) {
            topk_weights_out[k_idx] /= denom;
        }
    }
}

// Specialized version for 32 experts (1 thread per row, 32 elements per thread)
template <>
__device__ __forceinline__ void compute_topk_softmax_row<32, 1>(
    const __nv_bfloat16* __restrict__ router_logits_row,
    float* __restrict__ topk_weights_out,
    int32_t* __restrict__ topk_indices_out, int thread_group_idx, int lane_id,
    int warp_id) {
    constexpr int NUM_EXPERTS = 32;
    constexpr int topk = 4;
    constexpr int ELTS_PER_THREAD = NUM_EXPERTS;  // 32

    // Load all 32 router logits into registers
    float row_chunk[ELTS_PER_THREAD];

// Vectorized load (2 bfloat16 at a time)
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i += 2) {
        __nv_bfloat162 val =
            *reinterpret_cast<const __nv_bfloat162*>(&router_logits_row[i]);
        float2 f2 = __bfloat1622float2(val);
        row_chunk[i] = f2.x;
        row_chunk[i + 1] = f2.y;
    }

    // Step 1: Find max for numerical stability
    float thread_max = row_chunk[0];
#pragma unroll
    for (int i = 1; i < ELTS_PER_THREAD; i++) {
        thread_max = fmaxf(thread_max, row_chunk[i]);
    }

    // Step 2: Compute exp(x - max) and sum
    float row_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] = expf(row_chunk[i] - thread_max);
        row_sum += row_chunk[i];
    }

    // Step 3: Normalize to get softmax probabilities
    float reciprocal_sum = 1.0f / row_sum;
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
        row_chunk[i] *= reciprocal_sum;
    }

    // Step 4: Find top-4 values and indices using iterative argmax
    float selected_sum = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < topk; k_idx++) {
        // Find max in this thread's chunk
        float max_val = row_chunk[0];
        int expert = 0;

#pragma unroll
        for (int i = 1; i < ELTS_PER_THREAD; i++) {
            if (row_chunk[i] > max_val) {
                max_val = row_chunk[i];
                expert = i;
            }
        }

        // Store result
        topk_weights_out[k_idx] = max_val;
        topk_indices_out[k_idx] = expert;
        selected_sum += max_val;

        // Clear the winning value for next iteration
        if (k_idx + 1 < topk) {
            row_chunk[expert] = -10000.0f;
        }
    }

    // Step 5: Renormalize top-k weights to sum to 1.0
    float denom = (selected_sum > 0.0f) ? selected_sum : 1.0f;
#pragma unroll
    for (int k_idx = 0; k_idx < topk; k_idx++) {
        topk_weights_out[k_idx] /= denom;
    }
}

}  // namespace fused_routing

template <int NUM_EXPERTS, int topk, int NUM_BLOCK_SIZES, typename InValDtype,
          typename OutValDtype>
__global__ void fused_routing_kernel(
    const InValDtype* __restrict__ router_logits,  // [NUM_TOKENS, NUM_EXPERTS]
    const int64_t max_n_tiles, const int64_t NUM_TOKENS,
    OutValDtype* __restrict__ gate_scale, int32_t* __restrict__ topk_index,
    int32_t* __restrict__ gate_index, int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr,
    int32_t* __restrict__ prior_contrib, int32_t* __restrict__ hist_prefix,
    int padding_indices, int padding_weights) {
    TORCH_CHECK(false, "unimplemented kernel");
}

template <>
__global__ void fused_routing_kernel<128, 4, 4, __nv_bfloat16,
                                     __nv_bfloat16>(
    const __nv_bfloat16* __restrict__ router_logits,  // [NUM_TOKENS, 128]
    const int64_t max_n_tiles, const int64_t NUM_TOKENS,
    __nv_bfloat16* __restrict__ gate_scale, int32_t* __restrict__ topk_index,
    int32_t* __restrict__ gate_index, int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr,
    int32_t* __restrict__ _prior_contrib_, int32_t* __restrict__ _hist_prefix,
    int padding_indices, int padding_weights) {
    using namespace fused_routing;
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
    static constexpr int topk_padded = topk;
    static constexpr int NUM_BLOCK_SIZES = 4;
    int ROWS_PER_THREADS = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, num_threads);
    int ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, gridDim.x);
    int HYPO_ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, 8);
    static constexpr int THREAD_PER_CTA = 512;
    static constexpr int warp_size = 32;
    static constexpr int num_stages = 2;

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
    [2 * ROWS_PER_CTA * topk * sizeof(InValDtype)/sizeof(int32_t)] :
    topk_weights
    [2 * ROWS_PER_CTA * topk ] : topk_indices

    */
    using BlockScan = cub::BlockScan<int, 512>;
    __shared__ typename BlockScan::TempStorage temp_storage_hist;
    __shared__
        typename BlockScan::TempStorage temp_storage_tiles[NUM_BLOCK_SIZES];
    // __shared__ InValDtype sm_topk_weights[ROWS_PER_CTA * topk_padded];
    extern __shared__ int32_t __align__(16) sm_hist[];

    int topk_indices_sm_size = num_stages * topk_padded * THREAD_PER_CTA;
    int topk_weights_sm_size = num_stages * topk_padded * THREAD_PER_CTA *
                               sizeof(InValDtype) / sizeof(int32_t);
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
    size_t local_offset_size =
        FUSED_ROUTING_CEIL_DIV(ROWS_PER_CTA, warp_size) * warp_size * topk;
    int topk_indices_offset =
        local_offset_offset + local_offset_size + padding_indices;
    int topk_weights_offset =
        topk_indices_offset + topk_indices_sm_size + padding_weights;
    int shared_mem_size = topk_weights_offset + topk_weights_sm_size;
    __nv_bfloat16* topk_weights_ptr =
        reinterpret_cast<__nv_bfloat16*>(sm_hist + topk_weights_offset);
    int32_t* topk_indices_ptr = sm_hist + topk_indices_offset;

#pragma unroll
    for (int i = local_tid; i < shared_mem_size; i += blockDim.x) {
        sm_hist[i] = 0;
    }
    __syncthreads();  // Ensure all threads complete zeroing before setting
#pragma unroll
    for (int i = local_tid; i < NUM_BLOCK_SIZES * (max_n_tiles);
         i += blockDim.x) {
        sm_hist[block_pid_offset + i] = -1;
    }
    cluster.sync();  // we need to ensure global hist in CTA0 had been

    /*phase 0 + phase 1: Compute topk+softmax inline and build histograms*/
    int32_t* local_hist =
        reinterpret_cast<int32_t*>(sm_hist + local_hist_offset);
    int32_t* local_offset_sm =
        reinterpret_cast<int32_t*>(sm_hist + local_offset_offset);
    int row = CTA_ID * ROWS_PER_CTA;
    int row_end = min((int64_t)(row + ROWS_PER_CTA),
                      NUM_TOKENS);  // Each CTA only processes its own rows

    // Note: indices_ptr and weights_ptr were previously used for staging
    // in shared memory, but since Phase 0+1 now writes results directly to
    // global memory and Phase 3 reads from global memory, these are no
    // needed. The shared memory regions are kept for backward compatibility
    // with the memory layout.

    // For 128 experts, use 4 threads per row (32 elements each)
    // 512 threads = 128 rows per iteration (4 threads per row)
    constexpr int THREADS_PER_ROW_TOPK = 4;
    constexpr int ROWS_PER_ITER = THREAD_PER_CTA / THREADS_PER_ROW_TOPK;  //

    int lane_id = threadIdx.x % warp_size;
    int warp_id_local = threadIdx.x / warp_size;
    int thread_group_idx = threadIdx.x % THREADS_PER_ROW_TOPK;
    int row_in_iter = threadIdx.x / THREADS_PER_ROW_TOPK;

#pragma unroll 1
    for (int iter_base = row; iter_base < row_end; iter_base += ROWS_PER_ITER) {
        int current_row = iter_base + row_in_iter;
        int local_i = current_row - row;
        bool valid_row = (current_row < row_end);

        // Temporary storage for topk results
        float topk_weights_f[topk];
        int32_t topk_indices_local[topk];

        // CRITICAL: All threads must call compute_topk_softmax_row to avoid
        // deadlock. The function uses __shfl_xor_sync(0xffffffff, ...) which
        // requires all threads in the warp to participate. Invalid threads
        // use a safe address (row 0) to avoid out-of-bounds access.
        int safe_row = valid_row ? current_row : row;
        compute_topk_softmax_row<NUM_EXPERTS, THREADS_PER_ROW_TOPK>(
            router_logits + safe_row * NUM_EXPERTS, topk_weights_f,
            topk_indices_local, thread_group_idx, lane_id, warp_id_local);

        // Only valid threads write results
        if (valid_row && thread_group_idx == 0) {
// Write to global memory
#pragma unroll
            for (int k = 0; k < topk; k++) {
                topk_weights_ptr[local_i * topk + k] =
                    static_cast<InValDtype>(topk_weights_f[k]);
                topk_indices_ptr[local_i * topk + k] = topk_indices_local[k];
            }

            // Note: Shared memory writes for indices_ptr removed since
            // Phase 3 now reads directly from global memory.

// Build histogram and local offsets
#pragma unroll
            for (int k = 0; k < topk; k++) {
                int expert_id = topk_indices_local[k];
                if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
                    local_offset_sm[layout_addr(local_i, k)] =
                        atomicAdd(local_hist + expert_id, 1);
                }
            }
        }
        __syncthreads();  // Sync before next iteration
    }
    cluster.sync();
    int32_t* global_hist =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
    collect_hist<NUM_EXPERTS>(cluster, local_hist, global_hist);
    /* phase 2*/
    // compute expert_across_prefixsum, dst_experts[i] =
    // sum_{p=0}^{j-1}{local_hist[p]}, where i is expert_id, j is CTA_ID
    prefix_hist_CTA<NUM_EXPERTS>(cluster, sm_hist + local_hist_offset,
                                 sm_hist + expert_across_offset);
    cluster.sync();
    int h = 0;
    // Note: lane_id and warp_id_local already defined above for topk+softmax
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
    // topk_weights already written to global memory in Phase 0+1
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

    // Phase 3: Write gate_scale, topk_index, gate_index using the computed
    // results Read directly from global memory since Phase 0+1 already wrote
    // topk_weights and topk_indices there
#pragma unroll
    for (int i = row + local_tid; i < row_end; i += blockDim.x) {
        int local_i = i - row;

        // Load topk results from global memory
        for (int k = 0; k < topk; k++) {
            if (i >= 0 && i < NUM_TOKENS) {
                int expert_id = topk_indices_ptr[local_i * topk + k];
                if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
                    InValDtype val = topk_weights_ptr[local_i * topk + k];
                    int flat_idx = i * topk + k;
                    int expert_base = hist_sum_local[expert_id];
                    int expert_prior = prior_contrib[expert_id];
                    int expert_local = local_offset_sm[layout_addr(local_i, k)];

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
    const __nv_bfloat16* __restrict__ router_logits,  // [NUM_TOKENS, 32]
    const int64_t max_n_tiles, const int64_t num_tokens,
    __nv_bfloat16* __restrict__ gate_scale, int32_t* __restrict__ topk_index,
    int32_t* __restrict__ gate_index, int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr, int32_t* __restrict__ hist_ptr,
    int32_t* __restrict__ prior_contrib, int32_t* __restrict__ hist_prefix,
    int padding_indices, int padding_weights) {
    using namespace fused_routing;
    using InValDtype = __nv_bfloat16;
    using OutValDtype = __nv_bfloat16;
    namespace cg = cooperative_groups;
    auto NUM_TOKENS = num_tokens;
    auto MAX_N_TILES = max_n_tiles;
    auto PADDING_INDICES = padding_indices;
    auto PADDING_WEIGHTS = padding_weights;
    // cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    int local_tid = threadIdx.x;
    int tid = local_tid + blockDim.x * blockIdx.x;
    int CTA_ID = blockIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    static constexpr int NUM_EXPERTS = 32;
    static constexpr int topk = 4;
    static constexpr int topk_padded = topk;
    static constexpr int NUM_BLOCK_SIZES = 4;
    int ROWS_PER_THREADS = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, num_threads);
    int ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, gridDim.x);
    int HYPO_ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(NUM_TOKENS, 8);
    static constexpr int THREAD_PER_CTA = 512;
    static constexpr int warp_size = 32;
    static constexpr int num_stages = 2;

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
    extern __shared__ __align__(16) int32_t sm_hist[];
    int topk_indices_sm_size = FUSED_ROUTING_CEIL_DIV(ROWS_PER_CTA, warp_size) *
                               warp_size * topk_padded;
    int topk_weights_sm_size =
        topk_padded * ROWS_PER_CTA * sizeof(InValDtype) / sizeof(int32_t);
    int global_hist_offset = 0;
    int local_hist_offset = NUM_EXPERTS;
    int global_hist_exclusivesum_offset = local_hist_offset + NUM_EXPERTS;
    int token_offs_pad_offset =
        global_hist_exclusivesum_offset + NUM_EXPERTS + 1;
    int block_pid_offset =
        token_offs_pad_offset + (NUM_BLOCK_SIZES * (NUM_EXPERTS + 1));
    int expert_across_offset =
        block_pid_offset + (NUM_BLOCK_SIZES * MAX_N_TILES);
    int local_offset_offset = expert_across_offset + NUM_EXPERTS;
    size_t local_offset_size =
        FUSED_ROUTING_CEIL_DIV(ROWS_PER_CTA, warp_size) * warp_size * topk;
    int topk_indices_offset =
        local_offset_offset + local_offset_size + PADDING_INDICES;
    int topk_weights_offset =
        topk_indices_offset + topk_indices_sm_size + PADDING_WEIGHTS;
    int shared_mem_size = topk_weights_offset + topk_weights_sm_size;
    __nv_bfloat16* topk_weights_ptr =
        reinterpret_cast<__nv_bfloat16*>(sm_hist + topk_weights_offset);
    int32_t* topk_indices_ptr = sm_hist + topk_indices_offset;

#pragma unroll
    for (int i = local_tid; i < shared_mem_size; i += blockDim.x) {
        sm_hist[i] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = local_tid; i < NUM_BLOCK_SIZES * (MAX_N_TILES);
         i += blockDim.x) {
        sm_hist[block_pid_offset + i] = -1;
    }
    // cluster.sync();  // we need to ensure global hist in CTA0 had been
    // memset.
    grid.sync();

    /*phase 0 + phase 1: Compute topk+softmax inline and build histograms*/
    int32_t* local_hist =
        reinterpret_cast<int32_t*>(sm_hist + local_hist_offset);
    int32_t* local_offset_sm =
        reinterpret_cast<int32_t*>(sm_hist + local_offset_offset);
    int row = CTA_ID * ROWS_PER_CTA;
    int row_end = min((int64_t)(row + ROWS_PER_CTA),
                      NUM_TOKENS);  // Each CTA only processes its own rows

    // Note: indices_ptr removed since Phase 3 reads directly from global
    // memory.

    // For 32 experts, use 1 thread per row (each thread handles all 32 experts)
    // 512 threads = 512 rows per iteration
    constexpr int THREADS_PER_ROW_TOPK = 1;
    constexpr int ROWS_PER_ITER = THREAD_PER_CTA / THREADS_PER_ROW_TOPK;  // 512

    int lane_id = threadIdx.x % warp_size;
    int warp_id_local = threadIdx.x / warp_size;
    int thread_group_idx = 0;  // For 32 experts, each thread works alone
    int row_in_iter = threadIdx.x;
#pragma unroll 1
    for (int iter_base = row; iter_base < row_end; iter_base += ROWS_PER_ITER) {
        int current_row = iter_base + row_in_iter;
        int local_i = current_row - row;

        // Temporary storage for topk results
        float topk_weights_f[topk];
        int32_t topk_indices_local[topk];

        if (current_row < row_end) {
            // Compute topk+softmax for this row (each thread handles one row
            // independently)
            compute_topk_softmax_row<NUM_EXPERTS, THREADS_PER_ROW_TOPK>(
                router_logits + current_row * NUM_EXPERTS, topk_weights_f,
                topk_indices_local, thread_group_idx, lane_id, warp_id_local);

#pragma unroll
            for (int k = 0; k < topk; k++) {
                // (TODO)2 way bank conflict
                topk_weights_ptr[local_i * topk + k] =
                    static_cast<InValDtype>(topk_weights_f[k]);
                topk_indices_ptr[layout_addr(local_i, k)] =
                    topk_indices_local[k];
            }

            // Note: Shared memory writes for indices_ptr removed since
            // Phase 3 now reads directly from global memory.

// Build histogram and local offsets
#pragma unroll
            for (int k = 0; k < topk; k++) {
                int expert_id = topk_indices_local[k];
                local_offset_sm[layout_addr(local_i, k)] =
                    atomicAdd(local_hist + expert_id, 1);
            }
        }
        __syncthreads();  // Sync before next iteration
    }
    // cluster.sync();
    grid.sync();
    int32_t* global_hist =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
    _collect_hist<NUM_EXPERTS>(grid, local_hist, hist_ptr, global_hist);

    /*===========================================phase 2*/
    int warp_id = threadIdx.x / 32;
    // compute expert_across_prefixsum, dst_experts[i] =
    // sum_{p=0}^{j-1}{local_hist[p]}, where i is expert_id, j is CTA_ID
    // _prefix_hist_CTA<NUM_EXPERTS>(cluster, sm_hist + local_hist_offset,
    //                              sm_hist + expert_across_offset);
    _prefix_hist_CTA<NUM_EXPERTS>(grid, sm_hist + local_hist_offset,
                                  prior_contrib);
    // cluster.sync();
    grid.sync();
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
            warp_id * MAX_N_TILES;
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
    // topk_weights already written to global memory in Phase 0+1
    // cluster.sync();
    grid.sync();

    // WB global memory
    int token_offs_pad_size = NUM_BLOCK_SIZES * (NUM_EXPERTS + 1);
    int pid_map_size = NUM_BLOCK_SIZES * (MAX_N_TILES);

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
        // hist_ptr[local_tid] = global_hist_sm[local_tid];
        if (local_tid == 0) expt_offs_ptr[NUM_EXPERTS] = hist_sum[NUM_EXPERTS];
    }
    // cluster.sync();
    grid.sync();

    /*=============================================phase 3*/

    int32_t* prior_contrib_sm =
        reinterpret_cast<int32_t*>(sm_hist + expert_across_offset);
    if (local_tid < NUM_EXPERTS)
        prior_contrib_sm[local_tid] =
            prior_contrib[CTA_ID * NUM_EXPERTS + local_tid];

    /*
    WE HAVE TO SYNC TO CTA'S LOCAL SM SINCE MAP_SHARED_RANK LEADS TO
    cudaErrorLaunchFailure.
    */
    // int32_t* hist_sum_sm0 = cluster.map_shared_rank(
    //     reinterpret_cast<int32_t*>(sm_hist +
    //     global_hist_exclusivesum_offset), 0);
    // int32_t* hist_sum_local =
    //     reinterpret_cast<int32_t*>(sm_hist +
    //     global_hist_exclusivesum_offset);
    // if (local_tid < NUM_EXPERTS)
    //     hist_sum_local[local_tid] = hist_sum_sm0[local_tid];
    // cluster.sync();
    int32_t* hist_sum =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset);
    if (CTA_ID == 0 && local_tid <= NUM_EXPERTS)
        hist_prefix[local_tid] = hist_sum[local_tid];
    grid.sync();
    if (local_tid <= NUM_EXPERTS) hist_sum[local_tid] = hist_prefix[local_tid];

    grid.sync();

    // Phase 3: Write gate_scale, topk_index, gate_index using the computed topk
    // results Read directly from global memory since Phase 0+1 already wrote
    // topk_weights and topk_indices there
#pragma unroll
    for (int i = row + local_tid; i < row_end; i += blockDim.x) {
        int local_i = i - row;

        for (int k = 0; k < topk; k++) {
            int expert_id = topk_indices_ptr[layout_addr(local_i, k)];
            if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
                InValDtype val = topk_weights_ptr[local_i * topk + k];
                int flat_idx = i * topk + k;
                int expert_base = hist_sum[expert_id];
                int expert_prior = prior_contrib_sm[expert_id];
                int expert_local = local_offset_sm[layout_addr(local_i, k)];

                int global_pos = expert_base + expert_prior + expert_local;

                gate_scale[global_pos] = static_cast<OutValDtype>(val);
                topk_index[global_pos] = flat_idx;
                gate_index[flat_idx] = global_pos;
            }
        }
    }
}

}  // namespace moe
}  // namespace vllm

template <typename IdxType, typename InValType, typename OutValType>
void routing_kernel_helper(torch::Tensor& gating_output, int64_t max_n_tiles,
                           int64_t topk, torch::Tensor& gate_scale,
                           torch::Tensor& topk_index, torch::Tensor& gate_index,
                           torch::Tensor& token_offs_pad,
                           torch::Tensor& block_pid_map,
                           torch::Tensor& expt_offs, torch::Tensor& hist) {
    /*
    dispatch config
    */
    using namespace vllm::moe::fused_routing;
    TORCH_CHECK(topk == 4, "");
    constexpr int const_topk = 4;
    constexpr int const_topk_padded = const_topk;  // since we've had swizzle
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
            static constexpr int THREAD_PER_CTA = 512;

            int max_hw_limit = 0, dev = 0;
            cudaDeviceGetAttribute(
                &max_hw_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            int numSMs = deviceProp.multiProcessorCount;

            auto kernel_wrapper = &(
                vllm::moe::fused_routing_kernel<32, const_topk, NUM_BLOCK_SIZES,
                                                InValType, OutValType>);

            int padding_indices, padding_weights;
            size_t fixed_sm_size;
            int alignment_indices = 16;
            int alignment_weights = 8;

            auto compute_sm_for_rows = [&](size_t rows_per_cta) -> size_t {
                size_t global_hist_size = num_experts;
                size_t local_hist_size = num_experts;
                size_t global_hist_prefix_size = num_experts + 1;
                size_t token_offs_pad_size =
                    NUM_BLOCK_SIZES * (num_experts + 1);
                size_t block_pid_size = NUM_BLOCK_SIZES * (max_n_tiles);
                size_t prefix_experts_size = num_experts;
                size_t local_offset_size =
                    FUSED_ROUTING_CEIL_DIV(rows_per_cta, warp_size) *
                    warp_size * topk;
                size_t topk_weights_size = rows_per_cta * const_topk_padded;
                size_t topk_weights_size_int =
                    topk_weights_size * sizeof(InValType) / sizeof(int32_t);
                size_t topk_indices_size =
                    FUSED_ROUTING_CEIL_DIV(rows_per_cta, warp_size) *
                    warp_size * const_topk_padded;
                size_t topk_indices_size_int = topk_indices_size;

                fixed_sm_size = global_hist_size + local_hist_size +
                                global_hist_prefix_size + token_offs_pad_size +
                                block_pid_size + prefix_experts_size +
                                local_offset_size;

                size_t padding_before_indices_bytes = MAKE_ALIGNMENT_DIFF(
                    fixed_sm_size * sizeof(int32_t), alignment_indices);
                padding_indices = FUSED_ROUTING_CEIL_DIV(
                    padding_before_indices_bytes, sizeof(int32_t));

                size_t padding_before_weights_bytes = MAKE_ALIGNMENT_DIFF(
                    (fixed_sm_size + padding_indices + topk_indices_size) *
                        sizeof(int32_t),
                    alignment_weights);
                padding_weights = FUSED_ROUTING_CEIL_DIV(
                    padding_before_weights_bytes, sizeof(int32_t));

                size_t required_dynamicSmemBytes =
                    (fixed_sm_size + padding_indices + topk_indices_size_int +
                     padding_weights + topk_weights_size_int) *
                    sizeof(int32_t);
                return required_dynamicSmemBytes;
            };

            int grid_size = numSMs;
            size_t ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(num_tokens, grid_size);
            size_t smem_bytes = compute_sm_for_rows(ROWS_PER_CTA);

            while (smem_bytes > (size_t)max_hw_limit &&
                   grid_size < num_tokens) {
                grid_size *= 2;
                ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(num_tokens, grid_size);
                smem_bytes = compute_sm_for_rows(ROWS_PER_CTA);
            }
            TORCH_CHECK(smem_bytes <= (size_t)max_hw_limit,
                        "Shared memory requirement (", smem_bytes,
                        " bytes) exceeds hardware limit (", max_hw_limit,
                        " bytes)");

            constexpr int MAX_ITERATIONS = 5;
            for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
                auto cuda_error = cudaFuncSetAttribute(
                    kernel_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes);
                TORCH_CHECK(cuda_error == cudaSuccess,
                            "cudaFuncSetAttribute failed: ",
                            cudaGetErrorString(cuda_error));

                int numBlocksPerSM = 0;
                cuda_error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &numBlocksPerSM, kernel_wrapper, THREAD_PER_CTA,
                    smem_bytes);
                TORCH_CHECK(
                    cuda_error == cudaSuccess,
                    "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: ",
                    cudaGetErrorString(cuda_error));
                TORCH_CHECK(numBlocksPerSM > 0,
                            "Kernel cannot achieve any occupancy with smem=",
                            smem_bytes);

                int max_coop_grid = numSMs * numBlocksPerSM;
                int new_grid_size = std::min(max_coop_grid, (int)num_tokens);
                new_grid_size = std::max(new_grid_size, 1);

                if (new_grid_size == grid_size) {
#if KERNEL_DEBUG
                    std::cout << "Converged at iteration " << iter
                              << ", grid_size=" << grid_size << std::endl;
#endif
                    break;
                }

                grid_size = new_grid_size;
                ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(num_tokens, grid_size);
                smem_bytes = compute_sm_for_rows(ROWS_PER_CTA);

#if KERNEL_DEBUG
                std::cout << "Iteration " << iter << ": grid_size=" << grid_size
                          << ", ROWS_PER_CTA=" << ROWS_PER_CTA
                          << ", smem=" << smem_bytes / 1024.0 << " KB"
                          << ", numBlocksPerSM=" << numBlocksPerSM << std::endl;
#endif
            }

            ROWS_PER_CTA = FUSED_ROUTING_CEIL_DIV(num_tokens, grid_size);
            smem_bytes = compute_sm_for_rows(ROWS_PER_CTA);

            cudaFuncSetAttribute(kernel_wrapper,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_bytes);

#if KERNEL_DEBUG
            std::cout << "=== Final Launch Config ===" << std::endl;
            std::cout << "  num_tokens: " << num_tokens << std::endl;
            std::cout << "  grid_size: " << grid_size << std::endl;
            std::cout << "  ROWS_PER_CTA: " << ROWS_PER_CTA << std::endl;
            std::cout << "  smem_bytes: " << smem_bytes / 1024.0 << " KB"
                      << std::endl;
            std::cout << "  padding_indices: " << padding_indices << std::endl;
            std::cout << "  padding_weights: " << padding_weights << std::endl;
#endif

            // ========== Launch Kernel ==========
            auto grid_dim = dim3(grid_size, 1, 1);
            auto block_dim = dim3(THREAD_PER_CTA, 1, 1);
            const cudaStream_t current_stream =
                at::cuda::getCurrentCUDAStream();

            auto router_logits_ptr =
                reinterpret_cast<const InValType*>(gating_output.data_ptr());
            auto gate_scale_ptr =
                reinterpret_cast<OutValType*>(gate_scale.data_ptr());
            auto topk_index_ptr = topk_index.data_ptr<int32_t>();
            auto gate_index_ptr = gate_index.data_ptr<int32_t>();
            auto token_offs_pad_ptr = token_offs_pad.data_ptr<int32_t>();
            auto block_pid_map_ptr = block_pid_map.data_ptr<int32_t>();
            auto expt_offs_ptr = expt_offs.data_ptr<int32_t>();
            auto hist_ptr = hist.data_ptr<int32_t>();

            // Allocate temporary buffers for grid sync
            int32_t* prior_contrib;
            int32_t* hist_prefix;
            cudaMallocAsync(&prior_contrib,
                            grid_dim.x * num_experts * sizeof(int32_t),
                            current_stream);
            cudaMallocAsync(&hist_prefix,
                            grid_dim.x * num_experts * sizeof(int32_t),
                            current_stream);

            void* args[] = {
                (void*)&router_logits_ptr,  (void*)&max_n_tiles,
                (void*)&num_tokens,         (void*)&gate_scale_ptr,
                (void*)&topk_index_ptr,     (void*)&gate_index_ptr,
                (void*)&token_offs_pad_ptr, (void*)&block_pid_map_ptr,
                (void*)&expt_offs_ptr,      (void*)&hist_ptr,
                (void*)&prior_contrib,      (void*)&hist_prefix,
                (void*)&padding_indices,    (void*)&padding_weights};

            cudaLaunchCooperativeKernel((void*)kernel_wrapper, grid_dim,
                                        block_dim, args, smem_bytes,
                                        current_stream);
            // TORCH_CHECK(cuda_error == cudaSuccess,
            //             "cudaLaunchCooperativeKernel failed: ",
            //             cudaGetErrorString(cuda_error));

            // cudaStreamSynchronize(current_stream);
            cudaFreeAsync(prior_contrib, current_stream);
            cudaFreeAsync(hist_prefix, current_stream);
            break;
        }
        case 128: {
            auto warp_size = 32;
            int cluster_size = 0;
            static constexpr int THREAD_PER_CTA = 512;

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
                FUSED_ROUTING_CEIL_DIV(rows_per_cta, warp_size) * warp_size *
                topk;
            size_t topk_indices_size = 2 * THREAD_PER_CTA * const_topk_padded;
            size_t topk_indices_size_int = topk_indices_size;
            size_t topk_weights_size = 2 * THREAD_PER_CTA * const_topk_padded;
            size_t topk_weights_size_int =
                topk_weights_size * sizeof(InValType) / sizeof(int32_t);
            auto kernel_wrapper =
                &(vllm::moe::fused_routing_kernel<
                    128, const_topk, NUM_BLOCK_SIZES, InValType, OutValType>);

            cudaFuncAttributes attr;
            cudaLaunchConfig_t config = {};
            auto cuda_error = cudaFuncGetAttributes(&attr, kernel_wrapper);
            TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error));
            size_t static_smem_size = attr.sharedSizeBytes;

            size_t padding_indices, padding_weights, requried_sm_size,
                fixed_sm_size;
            int alignment_indices = 16;
            int alignment_weights = 8;
            auto compute_sm = [&]() {
                fixed_sm_size = global_hist_size + local_hist_size +
                                global_hist_prefix_size + token_offs_pad_size +
                                block_pid_size + prefix_experts_size +
                                local_offset_size;  // 4B unit
                // make sm size align to 16
                size_t padding_before_indices_bytes = MAKE_ALIGNMENT_DIFF(
                    fixed_sm_size * sizeof(int32_t), alignment_indices);

                padding_indices = FUSED_ROUTING_CEIL_DIV(
                    padding_before_indices_bytes, sizeof(int32_t));

                size_t padding_before_weights_bytes = MAKE_ALIGNMENT_DIFF(
                    (fixed_sm_size + padding_indices + topk_indices_size) *
                        sizeof(int32_t),
                    alignment_weights);
                padding_weights = FUSED_ROUTING_CEIL_DIV(
                    padding_before_weights_bytes, sizeof(int32_t));
                size_t required_dynamicSmemBytes =
                    (fixed_sm_size + padding_indices + topk_indices_size_int +
                     padding_weights + topk_weights_size_int) *
                    sizeof(int32_t);
                requried_sm_size = required_dynamicSmemBytes + static_smem_size;
                return required_dynamicSmemBytes;
            };
            config.dynamicSmemBytes = compute_sm();

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

            auto grid_dim = dim3(hypo_cluster_size, 1, 1);
            // recompute config
            if (cluster_size > hypo_cluster_size) {
                rows_per_cta = (num_tokens + cluster_size - 1) / cluster_size;
                local_offset_size =
                    FUSED_ROUTING_CEIL_DIV(rows_per_cta, warp_size) *
                    warp_size * topk;

                config.dynamicSmemBytes = compute_sm();
                cuda_error = cudaFuncSetAttribute(
                    kernel_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    config.dynamicSmemBytes);
                TORCH_CHECK(cuda_error == 0, cudaGetErrorString(cuda_error))
                grid_dim = dim3(cluster_size, 1, 1);
            }

#if KERNEL_DEBUG
            std::cout << "fixed sm size: " << fixed_sm_size << ","
                      << "padding weights: " << padding_before_weights
                      << std::endl;
#endif
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

            auto router_logits_ptr =
                reinterpret_cast<const InValType*>(gating_output.data_ptr());
            // auto topk_weights_ptr =
            //     reinterpret_cast<InValType*>(topk_weights.data_ptr());
            auto gate_scale_ptr =
                reinterpret_cast<OutValType*>(gate_scale.data_ptr());
            // auto topk_indices_ptr = topk_indices.data_ptr<IdxType>();
            auto topk_index_ptr = topk_index.data_ptr<int32_t>();
            auto gate_index_ptr = gate_index.data_ptr<int32_t>();
            auto token_offs_pad_ptr = token_offs_pad.data_ptr<int32_t>();
            auto block_pid_map_ptr = block_pid_map.data_ptr<int32_t>();
            auto expt_offs_ptr = expt_offs.data_ptr<int32_t>();
            auto hist_ptr = hist.data_ptr<int32_t>();

            int32_t* prior_contrib;
            int32_t* hist_prefix;
            cudaMalloc(&prior_contrib,
                       grid_dim.x * num_experts * sizeof(int32_t));
            cudaMalloc(&hist_prefix, (num_experts + 1) * sizeof(int32_t));
            cudaLaunchKernelEx(
                &config, kernel_wrapper, router_logits_ptr, max_n_tiles,
                num_tokens, gate_scale_ptr, topk_index_ptr, gate_index_ptr,
                token_offs_pad_ptr, block_pid_map_ptr, expt_offs_ptr, hist_ptr,
                prior_contrib, hist_prefix, padding_indices, padding_weights);
            break;
        }
            TORCH_CHECK(false, "Unsupported num experts: ", num_experts);
    }
}

void fused_routing(torch::Tensor& gating_output, int64_t max_n_tiles,
                   int64_t topk, torch::Tensor& gate_scale,
                   torch::Tensor& topk_index, torch::Tensor& gate_index,
                   torch::Tensor& token_offs_pad, torch::Tensor& block_pid_map,
                   torch::Tensor& expt_offs, torch::Tensor& hist) {
    /*
    dispatch dtype
    */
    if (gate_scale.scalar_type() == at::ScalarType::BFloat16) {
        // int int4_alignment = 16;
        // int float16_alignment = 8;
        // if ((reinterpret_cast<uintptr_t>(topk_indices.data_ptr()) %
        //      int4_alignment) != 0)
        //     TORCH_CHECK(false, "");
        // if ((reinterpret_cast<uintptr_t>(topk_weights.data_ptr()) %
        //      float16_alignment) != 0)
        //     TORCH_CHECK(false, "");
        routing_kernel_helper<int32_t, __nv_bfloat16, __nv_bfloat16>(
            gating_output, max_n_tiles, topk, gate_scale, topk_index,
            gate_index, token_offs_pad, block_pid_map, expt_offs, hist);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", gate_scale.scalar_type());
    }
}
