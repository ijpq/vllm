#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM 显存分析脚本 - 用于定位 GPT-OSS 20B 的显存瓶颈
使用方法: python memory_profile.py
"""

import gc
import sys
from dataclasses import dataclass

import torch


# 显存记录点
@dataclass
class MemorySnapshot:
    name: str
    allocated_mb: float
    reserved_mb: float
    peak_mb: float


snapshots: list[MemorySnapshot] = []


def take_snapshot(name: str):
    """记录当前显存状态"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2

    snap = MemorySnapshot(name, allocated, reserved, peak)
    snapshots.append(snap)
    print(
        f"[MEM] {name:40s} | Alloc: {allocated:8.1f} MB | Reserved: {reserved:8.1f} MB | Peak: {peak:8.1f} MB"
    )
    return snap


def print_memory_diff(name1: str, name2: str):
    """打印两个快照之间的差异"""
    s1 = next((s for s in snapshots if s.name == name1), None)
    s2 = next((s for s in snapshots if s.name == name2), None)
    if s1 and s2:
        diff = s2.allocated_mb - s1.allocated_mb
        print(f"[DIFF] {name1} → {name2}: {diff:+.1f} MB")


def print_summary():
    """打印显存使用摘要"""
    print("\n" + "=" * 80)
    print("显存使用摘要")
    print("=" * 80)

    if len(snapshots) < 2:
        print("快照数量不足")
        return

    # 找出最大增量
    max_diff = 0
    max_diff_pair = ("", "")
    for i in range(1, len(snapshots)):
        diff = snapshots[i].allocated_mb - snapshots[i - 1].allocated_mb
        if diff > max_diff:
            max_diff = diff
            max_diff_pair = (snapshots[i - 1].name, snapshots[i].name)

    print(f"\n最大显存增量: {max_diff:.1f} MB")
    print(f"发生在: {max_diff_pair[0]} → {max_diff_pair[1]}")

    print(f"\n最终显存使用: {snapshots[-1].allocated_mb:.1f} MB")
    print(f"峰值显存使用: {max(s.peak_mb for s in snapshots):.1f} MB")


# ============================================================
# 主测试流程
# ============================================================


def test_model_loading():
    """测试模型加载过程的显存使用"""

    print("\n" + "=" * 80)
    print("GPT-OSS 20B 显存分析")
    print("=" * 80 + "\n")

    # 清理并重置
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    take_snapshot("00_initial")

    # ---- Step 1: 导入 vLLM ----
    print("\n>>> 导入 vLLM...")
    from vllm import LLM, SamplingParams

    take_snapshot("01_import_vllm")

    # ---- Step 2: 创建 LLM 实例 ----
    print("\n>>> 创建 LLM 实例 (这会加载模型)...")

    # 使用最小配置
    try:
        llm = LLM(
            model="openai/gpt-oss-20b",  # 修改为你的模型路径
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,  # 尽量多用显存
            max_model_len=128,  # 最小上下文
            enforce_eager=True,  # 禁用 CUDA Graph
            disable_custom_all_reduce=True,
            # 其他可能的优化选项
        )
        take_snapshot("02_model_loaded")

        # ---- Step 3: 运行一次推理 ----
        print("\n>>> 运行测试推理...")
        output = llm.generate(["Hello"], SamplingParams(max_tokens=1))
        take_snapshot("03_after_inference")

    except torch.cuda.OutOfMemoryError:
        take_snapshot("XX_OOM")
        print("\n[ERROR] OOM 发生!")
        print(f"当前显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    print_summary()


def analyze_weight_loading():
    """分析权重加载过程的显存使用"""

    print("\n" + "=" * 80)
    print("权重加载详细分析")
    print("=" * 80 + "\n")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    take_snapshot("00_initial")

    # 手动加载模型配置
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("/data/models/gpt-oss-20b")
    take_snapshot("01_config_loaded")

    print("\n模型配置:")
    print(f"  hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  num_layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"  num_experts: {getattr(config, 'num_local_experts', 'N/A')}")
    print(f"  intermediate_size: {getattr(config, 'intermediate_size', 'N/A')}")

    # 计算理论权重大小
    try:
        h = config.hidden_size
        n_layers = config.num_hidden_layers
        n_experts = getattr(config, "num_local_experts", 1)
        inter = config.intermediate_size

        # MoE layers: w13 (gate+up) + w2 (down)
        moe_params = n_experts * (h * inter * 2 + inter * h)  # w13 + w2
        total_params = n_layers * moe_params

        fp4_size_gb = total_params * 0.5 / 1024**3
        print(f"\n理论 FP4 权重大小 (仅 MoE): {fp4_size_gb:.2f} GB")
    except Exception as e:
        print(f"无法计算理论大小: {e}")


def profile_marlin_workspace():
    """分析 Marlin workspace 的显存开销"""

    print("\n" + "=" * 80)
    print("Marlin Workspace 分析")
    print("=" * 80 + "\n")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    take_snapshot("00_initial")

    # 导入 marlin 工具
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        device = torch.device("cuda:0")

        # 测试不同大小的 workspace
        for num_experts in [1, 4, 8, 16]:
            ws = marlin_make_workspace_new(device, num_experts)
            size_mb = ws.numel() * ws.element_size() / 1024**2
            print(
                f"Workspace for {num_experts} experts: {size_mb:.2f} MB, shape={ws.shape}"
            )
            del ws

        take_snapshot("01_workspace_tested")

    except Exception as e:
        print(f"无法测试 workspace: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "weights":
            analyze_weight_loading()
        elif sys.argv[1] == "marlin":
            profile_marlin_workspace()
        else:
            print(f"未知选项: {sys.argv[1]}")
            print("用法: python memory_profile.py [weights|marlin]")
    else:
        test_model_loading()
