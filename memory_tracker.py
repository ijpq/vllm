# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
显存追踪工具 - 直接注入到 vLLM 代码中使用

使用方法:
1. 在 vLLM 代码开头添加: from memory_tracker import mem_track, print_mem_report
2. 在关键位置添加: mem_track("位置名称")
3. 最后调用: print_mem_report()
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import torch


@dataclass
class MemRecord:
    name: str
    allocated_mb: float
    reserved_mb: float
    timestamp: float
    delta_mb: float = 0.0


_records: list[MemRecord] = []
_last_allocated: float = 0.0


def mem_track(name: str) -> MemRecord:
    """记录当前显存状态"""
    global _last_allocated

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    delta = allocated - _last_allocated
    _last_allocated = allocated

    record = MemRecord(
        name=name,
        allocated_mb=allocated,
        reserved_mb=reserved,
        timestamp=datetime.now().timestamp(),
        delta_mb=delta,
    )
    _records.append(record)

    # 实时打印
    sign = "+" if delta >= 0 else ""
    print(f"[MEM] {name:50s} | {allocated:8.1f} MB | {sign}{delta:8.1f} MB")

    return record


def mem_track_func(name: str = None):
    """装饰器: 追踪函数执行前后的显存变化"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__qualname__
            mem_track(f"ENTER {func_name}")
            try:
                result = func(*args, **kwargs)
                mem_track(f"EXIT  {func_name}")
                return result
            except Exception:
                mem_track(f"ERROR {func_name}")
                raise

        return wrapper

    return decorator


def print_mem_report():
    """打印显存使用报告"""
    if not _records:
        print("没有记录的显存数据")
        return

    print("\n" + "=" * 80)
    print("显存使用报告")
    print("=" * 80)

    # 按增量排序找出最大开销
    sorted_records = sorted(_records, key=lambda r: r.delta_mb, reverse=True)

    print("\n### 显存增量 TOP 10 ###")
    print(f"{'位置':<50} | {'增量 (MB)':>12}")
    print("-" * 70)
    for r in sorted_records[:10]:
        if r.delta_mb > 0:
            print(f"{r.name:<50} | {r.delta_mb:>+12.1f}")

    print("\n### 总览 ###")
    print(f"最终显存: {_records[-1].allocated_mb:.1f} MB")
    print(f"峰值显存: {max(r.allocated_mb for r in _records):.1f} MB")
    print(f"总增量: {sum(r.delta_mb for r in _records):.1f} MB")


def reset():
    """重置记录"""
    global _records, _last_allocated
    _records = []
    _last_allocated = 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ============================================================
# 用于 Monkey Patch vLLM 的工具
# ============================================================


def patch_prepare_moe_fp4():
    """Patch prepare_moe_fp4_layer_for_marlin 以追踪显存"""

    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4

    original_func = marlin_utils_fp4.prepare_moe_fp4_layer_for_marlin

    def patched_func(layer, input_dtype=None):
        mem_track("prepare_moe_fp4: START")

        # 记录输入权重大小
        w13_size = layer.w13_weight.numel() * layer.w13_weight.element_size() / 1024**2
        w2_size = layer.w2_weight.numel() * layer.w2_weight.element_size() / 1024**2
        print(f"  w13_weight: {w13_size:.1f} MB, w2_weight: {w2_size:.1f} MB")

        result = original_func(layer, input_dtype)

        mem_track("prepare_moe_fp4: END")

        # 记录输出权重大小
        w13_size = layer.w13_weight.numel() * layer.w13_weight.element_size() / 1024**2
        w2_size = layer.w2_weight.numel() * layer.w2_weight.element_size() / 1024**2
        print(f"  w13_weight: {w13_size:.1f} MB, w2_weight: {w2_size:.1f} MB")

        return result

    marlin_utils_fp4.prepare_moe_fp4_layer_for_marlin = patched_func
    print("[PATCHED] prepare_moe_fp4_layer_for_marlin")


def patch_marlin_workspace():
    """Patch marlin_make_workspace_new 以追踪 workspace 分配"""

    from vllm.model_executor.layers.quantization.utils import marlin_utils

    original_func = marlin_utils.marlin_make_workspace_new
    _call_count = [0]

    def patched_func(device, num_experts=1):
        _call_count[0] += 1
        result = original_func(device, num_experts)
        size_mb = result.numel() * result.element_size() / 1024**2
        print(
            f"[WORKSPACE #{_call_count[0]}] experts={num_experts}, size={size_mb:.2f} MB"
        )
        mem_track(f"workspace_alloc_{_call_count[0]}")
        return result

    marlin_utils.marlin_make_workspace_new = patched_func
    print("[PATCHED] marlin_make_workspace_new")


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    print("显存追踪工具测试")

    if not torch.cuda.is_available():
        print("CUDA 不可用")
        exit(1)

    reset()
    mem_track("initial")

    # 分配一些显存测试
    x = torch.randn(1024, 1024, device="cuda")
    mem_track("after 4MB alloc")

    y = torch.randn(4096, 4096, device="cuda")
    mem_track("after 64MB alloc")

    del x
    torch.cuda.empty_cache()
    mem_track("after del x")

    del y
    torch.cuda.empty_cache()
    mem_track("after del y")

    print_mem_report()
