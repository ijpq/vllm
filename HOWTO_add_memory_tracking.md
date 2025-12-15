# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
在 prepare_moe_fp4_layer_for_marlin 函数开头添加这段代码：

位置: vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
函数: prepare_moe_fp4_layer_for_marlin (约第 223 行)
"""

# ============== 添加到函数开头 ==============


def _mem_gb():
    """获取当前显存使用 (GB)"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3


def _track(label):
    print(f"[MEM] {label}: {_mem_gb():.3f} GB")


# 在函数开头调用
_track(f"prepare_moe_fp4 START - layer={layer.__class__.__name__}")

# ============== 在各个关键步骤后添加 ==============

# 在 "layer.workspace = marlin_make_workspace_new(device, 4)" 后添加:
_track("after workspace alloc")

# 在 weight repack 循环后添加:
_track(f"after {name} repack")

# 在 scales 处理循环后添加:
_track(f"after {name} scales")

# 在函数末尾添加:
_track("prepare_moe_fp4 END")


# ============== 完整修改后的函数示例 ==============
"""
def prepare_moe_fp4_layer_for_marlin(
    layer: torch.nn.Module, input_dtype: torch.dtype | None = None
) -> None:
    # === 显存追踪 ===
    def _mem_gb():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**3
    def _track(label):
        print(f"[MEM] {label}: {_mem_gb():.3f} GB")
    
    _track("START prepare_moe_fp4")
    # === 显存追踪结束 ===
    
    logger.warning_once(...)
    
    is_nvfp4 = hasattr(layer, "w13_weight_scale_2")
    ...
    
    layer.workspace = marlin_make_workspace_new(device, 4)
    _track("after workspace")  # <-- 添加
    
    ...
    
    for name in ["w13_weight", "w2_weight"]:
        ...
        setattr(layer, name, weight)
        _track(f"after {name} repack")  # <-- 添加
    
    for name in ["w13", "w2"]:
        ...
        setattr(layer, name + "_weight_scale", scales)
        _track(f"after {name} scales")  # <-- 添加
    
    ...
    
    _track("END prepare_moe_fp4")  # <-- 添加
"""
