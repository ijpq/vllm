#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Worker 进程内显存追踪补丁

使用方法：
    python apply_memory_patch.py           # 应用补丁
    python apply_memory_patch.py --revert  # 还原补丁
"""

import os
import shutil
import sys
from pathlib import Path

VLLM_ROOT = "/home/tangke/nas/myvllm"

# 要追踪的函数和文件
PATCHES = [
    {
        "file": "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py",
        "func": "prepare_moe_fp4_layer_for_marlin",
        "marker": "# [MEM_TRACK] ",  # 用于识别补丁
    },
    {
        "file": "vllm/v1/worker/gpu_worker.py",
        "func": "load_model",
        "marker": "# [MEM_TRACK] ",
    },
]

# 追踪代码模板
TRACK_CODE = """
# [MEM_TRACK] === Memory Tracking Start ===
import torch as _torch
def _mem_track(label):
    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        allocated = _torch.cuda.memory_allocated() / 1024**3
        reserved = _torch.cuda.memory_reserved() / 1024**3
        peak = _torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM] {label:50s} | Alloc: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {peak:.2f} GB")
_mem_track("ENTER {func_name}")
# [MEM_TRACK] === Memory Tracking End ===
"""

TRACK_CODE_EXIT = """
# [MEM_TRACK] exit
_mem_track("EXIT  {func_name}")
"""


def find_function_def(lines, func_name):
    """找到函数定义的行号"""
    for i, line in enumerate(lines):
        if f"def {func_name}(" in line:
            return i
    return -1


def get_indent(line):
    """获取缩进"""
    return len(line) - len(line.lstrip())


def apply_patch(file_path, func_name, marker):
    """应用追踪补丁"""
    full_path = Path(VLLM_ROOT) / file_path

    if not full_path.exists():
        print(f"[SKIP] {file_path} not found")
        return False

    # 读取文件
    with open(full_path) as f:
        content = f.read()
        lines = content.split("\n")

    # 检查是否已经打补丁
    if marker in content:
        print(f"[SKIP] {file_path} already patched")
        return False

    # 备份
    backup_path = str(full_path) + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy(full_path, backup_path)
        print(f"[BACKUP] {backup_path}")

    # 找到函数
    func_line = find_function_def(lines, func_name)
    if func_line < 0:
        print(f"[ERROR] Function {func_name} not found in {file_path}")
        return False

    # 找到函数体开始（跳过def行和可能的docstring）
    body_start = func_line + 1
    indent = get_indent(lines[func_line]) + 4  # 函数体缩进

    # 跳过 docstring
    while body_start < len(lines):
        stripped = lines[body_start].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # 找到 docstring 结束
            quote = stripped[:3]
            if stripped.count(quote) >= 2:  # 单行 docstring
                body_start += 1
                break
            else:  # 多行 docstring
                body_start += 1
                while body_start < len(lines) and quote not in lines[body_start]:
                    body_start += 1
                body_start += 1
                break
        elif stripped == "":
            body_start += 1
        else:
            break

    # 生成追踪代码
    track_code = TRACK_CODE.format(func_name=func_name)
    track_lines = [
        " " * indent + line if line.strip() else ""
        for line in track_code.strip().split("\n")
    ]

    # 插入追踪代码
    new_lines = lines[:body_start] + track_lines + lines[body_start:]

    # 写回文件
    with open(full_path, "w") as f:
        f.write("\n".join(new_lines))

    print(f"[PATCHED] {file_path}:{func_name}")
    return True


def revert_patch(file_path):
    """还原补丁"""
    full_path = Path(VLLM_ROOT) / file_path
    backup_path = str(full_path) + ".bak"

    if os.path.exists(backup_path):
        shutil.copy(backup_path, full_path)
        print(f"[REVERTED] {file_path}")
        return True
    else:
        print(f"[SKIP] No backup for {file_path}")
        return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        print("=== Reverting patches ===")
        for patch in PATCHES:
            revert_patch(patch["file"])
    else:
        print("=== Applying memory tracking patches ===")
        for patch in PATCHES:
            apply_patch(patch["file"], patch["func"], patch["marker"])

        print("\n" + "=" * 60)
        print("补丁已应用！现在运行 vLLM，显存信息会打印到 worker 日志中。")
        print("还原命令: python apply_memory_patch.py --revert")
        print("=" * 60)


if __name__ == "__main__":
    main()
