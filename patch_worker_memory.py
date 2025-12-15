#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
为 vLLM v1 worker 添加显存追踪补丁

使用方法:
    python patch_worker_memory.py          # 应用补丁
    python patch_worker_memory.py --revert # 还原补丁
"""

import shutil
import sys
from pathlib import Path

WORKER_FILE = "/home/tangke/nas/myvllm/vllm/v1/worker/gpu_worker.py"

# 追踪代码 - 添加到文件开头的导入区域后
TRACKER_CODE = '''
# ========== [MEM_TRACK] 显存追踪代码开始 ==========
def _mem_track(label: str):
    """打印当前 GPU 显存使用情况"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM_TRACK] {label:40s} | Alloc: {allocated:6.2f} GB | Reserved: {reserved:6.2f} GB | Peak: {peak:6.2f} GB")
# ========== [MEM_TRACK] 显存追踪代码结束 ==========
'''

# 要在各个函数中添加的追踪点
PATCHES = [
    # load_model 函数
    {
        "find": "def load_model(self) -> None:",
        "after_docstring": True,
        "insert": '        _mem_track("load_model: START")',
    },
    {
        "find": "self.model_runner.load_model(eep_scale_up=eep_scale_up)",
        "after_line": True,
        "insert": '        _mem_track("load_model: END (after model_runner.load_model)")',
    },
    # determine_available_memory 函数
    {
        "find": "def determine_available_memory(self) -> int:",
        "after_docstring": True,
        "insert": '        _mem_track("determine_available_memory: START")',
    },
    {
        "find": "self.model_runner.profile_run()",
        "after_line": True,
        "insert": '            _mem_track("determine_available_memory: after profile_run")',
        "first_only": True,  # 只替换第一个匹配
    },
    # initialize_from_config 函数
    {
        "find": "def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:",
        "after_docstring": True,
        "insert": '        _mem_track("initialize_from_config: START")',
    },
]


def apply_patch():
    """应用显存追踪补丁"""
    path = Path(WORKER_FILE)

    if not path.exists():
        print(f"[ERROR] 文件不存在: {WORKER_FILE}")
        return False

    # 读取文件
    content = path.read_text()

    # 检查是否已打补丁
    if "[MEM_TRACK]" in content:
        print("[SKIP] 补丁已存在")
        return False

    # 备份
    backup_path = str(path) + ".bak"
    if not Path(backup_path).exists():
        shutil.copy(path, backup_path)
        print(f"[BACKUP] {backup_path}")

    lines = content.split("\n")

    # 1. 在导入区域后添加追踪函数
    # 找到最后一个 import 语句
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    # 在最后一个 import 后插入追踪代码
    lines.insert(last_import_idx + 1, TRACKER_CODE)

    content = "\n".join(lines)

    # 2. 在各个函数中添加追踪点
    for patch in PATCHES:
        find_str = patch["find"]
        insert_str = patch["insert"]

        if find_str not in content:
            print(f"[WARN] 未找到: {find_str[:50]}...")
            continue

        if patch.get("after_line"):
            # 在匹配行后插入
            if patch.get("first_only"):
                # 只替换第一个
                idx = content.find(find_str)
                if idx >= 0:
                    # 找到这一行的结尾
                    line_end = content.find("\n", idx)
                    content = (
                        content[: line_end + 1]
                        + insert_str
                        + "\n"
                        + content[line_end + 1 :]
                    )
            else:
                content = content.replace(find_str, find_str + "\n" + insert_str)
        elif patch.get("after_docstring"):
            # 在函数定义后（跳过 docstring）插入
            idx = content.find(find_str)
            if idx >= 0:
                # 找到函数定义行的结尾
                func_line_end = content.find("\n", idx)
                # 检查下一行是否是 docstring
                next_line_start = func_line_end + 1
                rest = content[next_line_start : next_line_start + 200]

                # 跳过 docstring
                if '"""' in rest[:50] or "'''" in rest[:50]:
                    # 找到 docstring 结束
                    quote = '"""' if '"""' in rest[:50] else "'''"
                    first_quote = rest.find(quote)
                    second_quote = rest.find(quote, first_quote + 3)
                    if second_quote > 0:
                        insert_pos = next_line_start + second_quote + 3
                        # 找到这一行的结尾
                        line_end = content.find("\n", insert_pos)
                        content = (
                            content[: line_end + 1]
                            + insert_str
                            + "\n"
                            + content[line_end + 1 :]
                        )
                    else:
                        # docstring 跨多行，简单处理：在函数定义后插入
                        content = (
                            content[: func_line_end + 1]
                            + insert_str
                            + "\n"
                            + content[func_line_end + 1 :]
                        )
                else:
                    # 没有 docstring，直接在函数定义后插入
                    content = (
                        content[: func_line_end + 1]
                        + insert_str
                        + "\n"
                        + content[func_line_end + 1 :]
                    )

    # 写回文件
    path.write_text(content)
    print(f"[PATCHED] {WORKER_FILE}")
    print("\n显存追踪点已添加到:")
    print("  - load_model()")
    print("  - determine_available_memory()")
    print("  - initialize_from_config()")
    return True


def revert_patch():
    """还原补丁"""
    path = Path(WORKER_FILE)
    backup_path = Path(str(path) + ".bak")

    if backup_path.exists():
        shutil.copy(backup_path, path)
        print(f"[REVERTED] {WORKER_FILE}")
        return True
    else:
        print(f"[ERROR] 备份文件不存在: {backup_path}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        revert_patch()
    else:
        apply_patch()
