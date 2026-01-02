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
template <int NUM_EXPERTS, int topk>
__global__ void fused_routing_kernel(
    float* __restrict__ router_logits, float* __restrict__ topk_weights,
    int16_t* __restrict__ topk_indices, int64_t max_n_tiles, int64_t NUM_TOKENS,
    float* __restrict__ gate_scale, int16_t* __restrict__ topk_index,
    int32_t* __restrict__ gate_index, int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr);

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
__global__ void fused_routing_kernel<32, 4>(
    float* __restrict__ router_logits, float* __restrict__ topk_weights,
    int16_t* __restrict__ topk_indices, int64_t max_n_tiles, int64_t NUM_TOKENS,
    float* __restrict__ gate_scale, int16_t* __restrict__ topk_index,
    int32_t* __restrict__ gate_index, int32_t* __restrict__ token_offs_pad_ptr,
    int32_t* __restrict__ block_pid_map_ptr,
    int32_t* __restrict__ expt_offs_ptr) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  constexpr int NUM_BLOCK_SIZES = 4;
  int local_tid = threadIdx.x;
  int tid = local_tid + blockDim.x * blockIdx.x;
  const int NUM_EXPERTS = 32;
  const int topk = 4;

  // shared memory layout
  extern __shared__ int32_t sm_hist[];
  int global_hist_offset = 0;
  int local_hist_offset = NUM_EXPERTS;
  int global_hist_exclusivesum_offset = local_hist_offset + NUM_EXPERTS;
  int token_offs_pad_offset = global_hist_exclusivesum_offset + NUM_EXPERTS + 1;
  int block_pid_offset =
      token_offs_pad_offset + (NUM_BLOCK_SIZES * (NUM_EXPERTS + 1));
  int expert_across_offset = block_pid_offset + (NUM_BLOCK_SIZES * max_n_tiles);
  int shared_mem_size = expert_across_offset + NUM_EXPERTS;
/*
Assumingly, shared Mem is organized as this layout:
[NUM_EXPERTS] : global hist
[NUM_EXPERTS] : local histgram for each threadblock
[NUM_EXPERTS+1]: global hist exclusive-sum
[NUM_BLOCK_SIZES, NUM_EXPERTS+1] : token_offs_pad
[NUM_BLOCK_SIZES, max_n_tiles] : block_pid
[NUM_EXPERTS]: exclusivesum for experts
*/
#pragma unroll
  for (int i = local_tid; i < shared_mem_size; i += blockDim.x) {
    sm_hist[i] = 0;
  }
  // __syncthreads();
  cluster.sync();  // we need to ensure global hist had been memset.

  /*phase 1*/
  int my_local_offset[topk];
  // compute local histgram, sync to global hist.
  int32_t* local_hist = reinterpret_cast<int32_t*>(sm_hist + local_hist_offset);
  if (tid < NUM_TOKENS) {
    int64_t row_experts =
        *reinterpret_cast<int64_t*>(topk_indices + tid * topk);
    auto expt0 = static_cast<int32_t>(row_experts & 0xFFFF);
    auto expt1 = static_cast<int32_t>(row_experts >> 16 & 0xFFFF);
    auto expt2 = static_cast<int32_t>(row_experts >> 32 & 0xFFFF);
    auto expt3 = static_cast<int32_t>(row_experts >> 48 & 0xFFFF);
    my_local_offset[0] = atomicAdd(local_hist + expt0, 1);
    my_local_offset[1] = atomicAdd(local_hist + expt1, 1);
    my_local_offset[2] = atomicAdd(local_hist + expt2, 1);
    my_local_offset[3] = atomicAdd(local_hist + expt3, 1);
  }
  __syncthreads();
  int32_t* global_hist = cluster.map_shared_rank(
      reinterpret_cast<int32_t*>(sm_hist + global_hist_offset), 0);
#pragma unroll
  for (int i = threadIdx.x; i < NUM_EXPERTS; i += blockDim.x) {
    atomicAdd(global_hist + i, local_hist[i]);
  }
  cluster.sync();

  /* phase 2*/
  if (blockIdx.x == 0 && local_tid < 32) {
    // compute expert_across_prefixsum, dst_experts[i] =
    // sum_p_0_j-1(local_hist[p]), where i is expert_id, j is tb's id
    // XXX(ijpq): we may leverage warpscan to improve this process
    using WarpScan = cub::WarpScan<int>;
    __shared__ typename WarpScan::TempStorage temp_storage[1];
    for (int bidx = 1; bidx < gridDim.x; bidx++) {
      int32_t* dst_experts = cluster.map_shared_rank(
          reinterpret_cast<int32_t*>(sm_hist + expert_across_offset), bidx);
      int32_t* tb_local_hist = cluster.map_shared_rank(
          reinterpret_cast<int32_t*>(sm_hist + local_hist_offset), bidx - 1);
      int32_t* accum_hist = cluster.map_shared_rank(
          reinterpret_cast<int32_t*>(sm_hist + expert_across_offset), bidx - 1);
      dst_experts[local_tid] = tb_local_hist[local_tid] + accum_hist[local_tid];
    }

    int h = global_hist[local_tid];  // how many tokens are routed to this
                                     // expert, globally. let's say [100,20,50]
    int exclusive_res = 0;
    int warp_id = threadIdx.x / 32;
    int warp_reduce = 0;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(h, exclusive_res, warp_reduce);
    __syncwarp();  // to ensure exclusive sum finished within one warp

    // compute global hist exclusive sum
    // we get exclusive sum  [0, 100,120, 170]
    int32_t* hist_sum =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset);
    hist_sum[local_tid] = exclusive_res;
    hist_sum[32] = warp_reduce;

    // align the data with triton's matmul_ogs
    int BLOCK_M_LOG2_START = 4;
    for (int size = 0; size < NUM_BLOCK_SIZES; size++) {
      int block_m_log2 = BLOCK_M_LOG2_START + size;
      int block_m = 1 << block_m_log2;  // block_m = 16, 32,64,128
      int32_t* pid_map_row =
          reinterpret_cast<int32_t*>(sm_hist + block_pid_offset);

      int n_tiles =
          (h + block_m - 1) /
          block_m;  // group num_tokens into tiles, [7, 1, 8] when block_m = 16
      int warp_reduce = 0;
      WarpScan(temp_storage[warp_id])
          .ExclusiveSum(n_tiles, exclusive_res, warp_reduce);
      __syncwarp();
      // we get exclusive sum  [0, 7, 8, 16]

      // compute tiles exclusive sum
      int32_t* token_offs_pad =
          reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset);
      token_offs_pad[size * (NUM_EXPERTS + 1) + local_tid] = exclusive_res;
      if (local_tid == 0)
        token_offs_pad[size * (NUM_EXPERTS + 1) + 32] = warp_reduce;

      int tile_start = token_offs_pad[size * (NUM_EXPERTS + 1) + local_tid];
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
        int packed_val = (block_idx << 16) | local_tid;
        pid_map_row[(tile_start + block_idx)] = packed_val;
      }
    }
  }

  cluster.sync();
  // WB global memory(pidmap, tokenoffspad, global hist sum)
  int token_offs_pad_size = NUM_BLOCK_SIZES * (NUM_EXPERTS + 1);
  int pid_map_size = NUM_BLOCK_SIZES * (max_n_tiles);
  int32_t* token_offs_pad = cluster.map_shared_rank(
      reinterpret_cast<int32_t*>(sm_hist + token_offs_pad_offset), 0);
  int32_t* block_pid = cluster.map_shared_rank(
      reinterpret_cast<int32_t*>(sm_hist + block_pid_offset), 0);

  if (local_tid < 32) {
    int index = local_tid + 32 * blockIdx.x;
    if (index < token_offs_pad_size)
      token_offs_pad_ptr[index] = token_offs_pad[index];
    // TODO(ijpq): pad blockpidmap to avoid conflict
    if (index < pid_map_size) block_pid_map_ptr[index] = block_pid[index];

    int32_t* hist_sum =
        reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset);
    if (blockIdx.x == 0 && local_tid < 32)
      expt_offs_ptr[local_tid] = hist_sum[local_tid];
    if (blockIdx.x == 0 && local_tid == 0)
      expt_offs_ptr[NUM_EXPERTS] = hist_sum[NUM_EXPERTS];
  }

  /* phase 3*/

  int32_t* hist_sum = cluster.map_shared_rank(
      reinterpret_cast<int32_t*>(sm_hist + global_hist_exclusivesum_offset), 0);
  if (tid < NUM_TOKENS) {
    int32_t* prior_contrib =
        reinterpret_cast<int32_t*>(sm_hist + expert_across_offset);
    int topk_idx_stride = topk, topk_val_stride = topk;
    for (int i = 0; i < topk; i++) {
      int expert_id = topk_indices[tid * topk_idx_stride + i];
      int val = topk_weights[tid * topk_val_stride + i];
      int flat_idx = tid * topk + i;
      int expert_base = hist_sum[expert_id];
      int expert_prior = prior_contrib[expert_id];
      int expert_local = my_local_offset[i];
      int global_pos = expert_base + expert_prior + expert_local;

      gate_scale[global_pos] = val;
      topk_index[global_pos] = flat_idx;
      gate_index[flat_idx] = global_pos;
    }
  }
}

void fused_routing(torch::Tensor& gating_output, torch::Tensor& topk_weights,
                   torch::Tensor& topk_indices, int64_t max_n_tiles,
                   int64_t topk, torch::Tensor& gate_scale,
                   torch::Tensor& topk_index, torch::Tensor& gate_index,
                   torch::Tensor& token_offs_pad, torch::Tensor& block_pid_map,
                   torch::Tensor& expt_offs) {
  TORCH_CHECK(topk == 4, "");
  constexpr int const_topk = 4;
  auto grid_dim = dim3(32, 1, 1);
  auto block_dim = dim3(32, 1, 1);
  const auto num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  switch (num_experts) {
    case 32: {
      cudaLaunchConfig_t config = {0};
      // The grid dimension is not affected by cluster launch, and is still
      // enumerated using number of blocks. The grid dimension should be a
      // multiple of cluster size.
      config.gridDim = grid_dim;
      config.blockDim = block_dim;

      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = 32;  // Cluster size in X-dimension
      attribute[0].val.clusterDim.y = 1;
      attribute[0].val.clusterDim.z = 1;
      config.attrs = attribute;
      config.numAttrs = 1;

      auto gating_output_ptr = gating_output.data_ptr<float>();
      auto topk_weights_ptr = topk_weights.data_ptr<float>();
      auto topk_indices_ptr = topk_indices.data_ptr<int16_t>();
      auto gate_scale_ptr = gate_scale.data_ptr<float>();
      auto topk_index_ptr = topk_index.data_ptr<int16_t>();
      auto gate_index_ptr = gate_index.data_ptr<int32_t>();
      auto token_offs_pad_ptr = token_offs_pad.data_ptr<int32_t>();
      auto block_pid_map_ptr = block_pid_map.data_ptr<int32_t>();
      auto expt_offs_ptr = expt_offs.data_ptr<int32_t>();
      cudaLaunchKernelEx(&config, fused_routing_kernel<32, const_topk>,
                         gating_output_ptr, topk_weights_ptr, topk_indices_ptr,
                         max_n_tiles, num_tokens, gate_scale_ptr,
                         topk_index_ptr, gate_index_ptr, token_offs_pad_ptr,
                         block_pid_map_ptr, expt_offs_ptr);
      break;
    }
    case 128: {
      break;
    }
    default: {
      TORCH_CHECK(false, "Unsupported num experts: ", num_experts);
    }
  }
}
}  // namespace moe
}  // namespace vllm