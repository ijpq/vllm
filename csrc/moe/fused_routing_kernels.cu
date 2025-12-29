#include <type_traits>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../cuda_compat.h"
#include "../cub_helpers.h"

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
template<int NUM_EXPERTS>
void fused_routing_kernel();

__global__
template<>
void  fused_routing_kernel(__restrict__ float* router_logits, __restrict__ float* topk_weights, __restrict__ int16_t* topk_indices, __restrict__ int32_t* hist_ptr,  __restrict__ int32_t* partial_hist_ptr,
  __restrict__ int32_t* expt_offs_ptr,
 int topk)<128> {
 }

__global__
template<>
void  fused_routing_kernel(__restrict__ float* router_logits, __restrict__ float* topk_weights, __restrict__ int16_t* topk_indices, __restrict__ int32_t* hist_ptr,  __restrict__ int32_t* partial_hist_ptr,
  __restrict__ int32_t* expt_offs_ptr,
 int topk)<32> {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  // static_assert(NUM_EXPERTS == 32 || NUM_EXPERTS == 128);

  constexpr int NUM_BLOCK_SIZES =4;

  int local_tid = threadIdx.x;
  int tid = local_tid + blockDim.x * blockIdx.x;
  
  extern __shared__ int32_t sm_hist[];
    int global_hist_offset = NUM_EXPERTS, local_hist_offset = NUM_EXPERTS, global_hist_exclusivesum_offset = NUM_EXPERTS+1, token_offs_pad_offset = (NUM_BLOCK_SIZES * (NUM_EXPERTS+1));
  /*
  Assumingly, shared Mem is organized as this layout:
  [NUM_EXPERTS] : global hist 
  [NUM_EXPERTS] : local histgram for each threadblock
  [NUM_EXPERTS+1]: global hist exclusive-sum
  [NUM_BLOCK_SIZES, NUM_EXPERTS+1] : token_offs_pad
  [NUM_BLOCK_SIZES, max_n_tiles] : block_pid
  */
  #pragma unroll
  for (int i = local_tid; i < NUM_EXPERTS; i += blockDim.x) {
    sm_hist[i] = 0;
  }
  // __syncthreads();  
  cluster.sync(); // we need to ensure global hist had been memset.

  /*phase 1*/
  int start_row_idx = blockIdx.x, row_stride = blockDim.x;
  // int num_experts_4B = NUM_EXPERTS / sizeof(int32_t);
  int32_t* local_hist = reinterpret_cast<int32_t*>(sm_hist + NUM_EXPERTS); 
  if (tid < NUM_TOKENS) {
    int64_t row_experts = *reinterpret_cast<int64_t*>(topk_indices + tid * topk);
    atmoicAdd(local_hist + reinterpret_cast<int32_t>(row_experts & 0xFFFF), 1);
    atmoicAdd(local_hist + reinterpret_cast<int32_t>(row_experts >> 16 & 0xFFFF), 1);
    atmoicAdd(local_hist + reinterpret_cast<int32_t>(row_experts >> 32 & 0xFFFF), 1);
    atmoicAdd(local_hist + reinterpret_cast<int32_t>(row_experts >> 48 & 0xFFFF), 1);
    // for (int i =0 ;i < topk; i++) {
    //   int stride_m = topk * sizeof(int16_t);
    //   int expert_id = topk_indices[tid * stride_m + i];
    //   local_hist[expert_id]++;
      // int gid = topk_indices[tid * (topk * sizeof(int16_t))+ i] / 32;
      // int offs = topk_indices[tid* (topk* sizeof(int16_t)) + i] % 32;
      // local_hist += 1 << (gid * 32 + offs);
    // }

  }
  __syncthreads();
  //XXX(ijpq): write multiple global hist in each tb's sm leads to more insts.
  int32_t* global_hist = cluster.map_shared_rank(reinterpret_cast<int32_t*>(sm_hist), 0);
  #pragma unroll
  for (int i = threadIdx.x; i < NUM_EXPERTS; i += blockDim.x) {
        atomicAdd(global_hist[i], local_hist[i]);
  }

  cluster.sync();

  /* phase 2*/
  using WarpScan = cub::WarpScan<int>;
  __shared__ typename WarpScan::TempStorage temp_storage[1];
  if (blockIdx.x == 0  && local_tid < 32) {
    
    int h = global_hist[local_tid]; // how many tokens are routed to this expert, globally. let's say [100,20,50]
    int exclusive_res = 0;
    int warp_id = threadIdx.x / 32;
    int warp_reduce = 0;
    WarpScan(temp_storage[warp_id]).ExclusiveSum(h, exclusive_res, warp_reduce);
    __syncwarp(); // to ensure exclusive sum finished within one warp

    // we get exclusive sum , [0, 100,120, 170]
    int32_t* hist_sum = reinterpret_cast<int32_t*>(sm_hist + global_hist_offset);
    hist_sum[local_tid] = exclusive_res;
    hist_sum[32] = warp_reduce;
    
    // align the data with triton's matmul_ogs
    for (int size = 0 ; size < NUM_BLOCK_SIZES; size++) {
      int block_m_log2 = BLOCK_M_LOG2_START + size;
      int block_m = 1 << block_m_log2; // block_m = 16, 32,64,128
      int32_t* pid_map_row_ptr = reinterpret_cast<int32_t*>(sm_hist + global_hist_offset + local_hist_offset + expt_offs_offset + token_offs_pad_offset + size * max_n_tiles);

      // int h = hist_ptr[local_tid]; 
      int n_tiles = (h + block_m -1) / block_m; // group num_tokens into tiles, [7, 1, 8] when block_m = 16
      int warp_reduce = 0;
      WarpScan(temp_storage[warp_id]).ExclusiveSum(n_tiles,exclusive_res,warp_reduce);
      __syncwarp();
      // we get exclusive sum , [0, 7, 8, 16]

      int32_t* token_offs_pad = reinterpret_cast<int32_t*>(sm_hist + NUM_EXPERTS*2 +1);
      token_offs_pad_ptr[size * (NUM_EXPERTS+1) + local_tid] = exclusive_res;
      token_offs_pad_ptr[size * (NUM_EXPERTS+1) + 32] = warp_reduce;
      // __syncwarp();

      int tile_start = token_offs_pad_ptr[size * (NUM_EXPERTS+1) +  local_tid];
      /*
      tid0: h = 100, n_tiles = 7; tid1: h=20, n_tiles= 1; tid2: h = 50, n_tiles=8;

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
        pid_map_row_ptr[(tile_start + block_idx)] = packed_val;
      }

  }
  
  
}

  cluster.sync();

  /* phase 3*/
  
  if (tid < NUM_TOKENS) {
    int topk_idx_stride =  topk_val_stride = topk;
    for (int i = 0; i < topk; i++) {
      int expert_id = topk_indices[tid * topk_idx_stride + i];
      int val = topk_weights[tid * topk_val_stride + i];
      int flat_idx = tid * topk + i;
      int expert_base = expt_offs

    }
  }


 }



void fused_routing(torch::Tensor& gating_output, torch::Tensor& topk_weights,
                   torch::Tensor& topk_indices, torch::Tensor& hist,
                   torch::Tensor& expt_offs, torch::Tensor& partial_hist,
                   torch::Tensor& gate_scale, torch::Tensor& topk_index,
                   torch::Tensor& gate_index, torch::Tensor& token_offs_pad,
                   torch::Tensor& block_pid_map, int64_t max_n_tiles,
                    constexpr int64_t topk, bool renormalize = true) {
  static_assert(topk == 4 , "");
  auto grid_dim =;
   auto block_dim = ;
    const int num_experts = gating_output.size(-1);
    const auto num_tokens = gating_output.numel() / num_experts;
    switch (num_experts) : {

      case 32:
        break;
      case 128:
        break;
        TORCH_CHECK(false, "Unsupported num experts: ", num_experts);
      
    }
    cudaLaunchKernelEx();
  //  fused_routing_kernel<<<grid_dim, block_dim,>>>(gating_output.data_ptr<float>(), );
}
}  // namespace moe
}  // namespace vllm