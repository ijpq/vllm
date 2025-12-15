# GPT-OSS 20B 显存优化调查清单

## 背景
- OpenAI 声称 16GB 显存可运行
- vLLM 目前无法在 16GB 显存上运行
- 目标: 找出 vLLM 的额外显存开销并优化

---

## 1. 显存组成分析

### 1.1 理论最小显存 (~11.3 GB)
```
权重 (FP4):     20B × 0.5 bytes  = 10.0 GB
Scales (FP8):   20B/16 × 1 byte  =  1.25 GB
Global Scales:  微量              =  0.01 GB
-----------------------------------------
合计:                              11.26 GB
```

### 1.2 vLLM 额外开销 (需要测量)
- [ ] KV Cache 预分配
- [ ] Marlin Workspace (每层 vs 共享?)
- [ ] 权重转换临时副本
- [ ] CUDA Context
- [ ] Activation 缓存

---

## 2. 调查检查点

### 2.1 权重加载阶段
```python
# 在 prepare_moe_fp4_layer_for_marlin 中添加监控
import torch

def prepare_moe_fp4_layer_for_marlin(...):
    print(f"[BEFORE] GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # ... 原有代码 ...
    
    # 检查 gptq_marlin_repack 是否创建副本
    qweight = weight[i].view(torch.int32).T.contiguous()  # <- 这里可能创建副本!
    print(f"[AFTER repack] GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

**检查点:**
- [ ] `weight[i].view(torch.int32).T.contiguous()` 是否创建了临时副本?
- [ ] `ops.gptq_marlin_repack()` 是否 in-place 操作?
- [ ] 转换后原始权重是否被释放?

### 2.2 Marlin Workspace
```bash
# 检查 workspace 大小
grep -n "marlin_make_workspace" vllm/model_executor/layers/quantization/utils/marlin_utils.py
```

**检查点:**
- [ ] 每个 MoE 层分配独立 workspace 还是共享?
- [ ] workspace 大小是多少 MB?
- [ ] 能否改为全局共享?

### 2.3 KV Cache
```bash
# 检查 KV Cache 配置
grep -rn "num_gpu_blocks\|num_cpu_blocks" vllm/
```

**检查点:**
- [ ] KV Cache 预分配了多少 blocks?
- [ ] 每个 block 多大?
- [ ] `gpu_memory_utilization` 如何影响分配?

### 2.4 Activation Buffers
```bash
# 检查 MoE 层的临时分配
grep -n "torch.empty\|torch.zeros" vllm/model_executor/layers/fused_moe/
```

---

## 3. 快速测试命令

### 3.1 最小配置启动
```bash
python -c "
from vllm import LLM
llm = LLM(
    model='/path/to/gpt-oss-20b',
    gpu_memory_utilization=0.98,
    max_model_len=32,
    enforce_eager=True,
)
"
```

### 3.2 显存监控
```bash
# 另一个终端
watch -n 0.5 nvidia-smi
```

### 3.3 详细显存追踪
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False python your_script.py
```

---

## 4. 优化候选方案

### 4.1 高优先级 (预计收益大)
| 方案 | 预估节省 | 难度 | 状态 |
|------|----------|------|------|
| 共享 Marlin Workspace | ~1 GB | 中 | [ ] |
| In-place 权重转换 | ~2-4 GB | 高 | [ ] |
| 延迟 KV Cache 分配 | ~1-2 GB | 中 | [ ] |

### 4.2 中优先级
| 方案 | 预估节省 | 难度 | 状态 |
|------|----------|------|------|
| 减少 block 预分配 | ~0.5 GB | 低 | [ ] |
| 禁用不必要 buffers | ~0.3 GB | 低 | [ ] |

### 4.3 低优先级
| 方案 | 预估节省 | 难度 | 状态 |
|------|----------|------|------|
| CUDA Graph 内存优化 | ~0.2 GB | 高 | [ ] |

---

## 5. 代码修改位置

### 5.1 权重转换
```
vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
  └── prepare_moe_fp4_layer_for_marlin()  # 主要修改点
```

### 5.2 Workspace 分配
```
vllm/model_executor/layers/quantization/utils/marlin_utils.py
  └── marlin_make_workspace_new()
```

### 5.3 KV Cache
```
vllm/worker/worker.py
  └── _init_cache_engine()
  
vllm/core/block_manager.py
  └── BlockAllocator
```

---

## 6. 日志记录

### Day 1: ____
- 测量结果:
- 发现问题:

### Day 2: ____
- 测量结果:
- 发现问题:

---

## 7. 参考资料

- OpenAI GPT-OSS 博客: [URL]
- vLLM Memory Management: https://docs.vllm.ai/
- Marlin Kernel Paper: [URL]
