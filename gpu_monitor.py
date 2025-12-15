#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
使用 pynvml 监控 GPU 显存（跨进程）
"""

import threading
import time
from datetime import datetime

import pynvml


class GPUMemoryMonitor:
    def __init__(self, gpu_id=0, interval=0.5):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval
        self.records = []
        self.running = False
        self.thread = None

    def _get_memory_mb(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.used / 1024**2

    def _monitor_loop(self):
        while self.running:
            mem = self._get_memory_mb()
            self.records.append((datetime.now(), mem))
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.records = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[Monitor] Started, interval={self.interval}s")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[Monitor] Stopped, {len(self.records)} samples collected")

    def mark(self, label):
        """标记当前时间点"""
        mem = self._get_memory_mb()
        print(f"[MARK] {label}: {mem:.1f} MB")
        self.records.append((datetime.now(), mem, label))

    def report(self):
        if not self.records:
            print("No data")
            return

        # 找出峰值
        max_mem = max(r[1] for r in self.records)
        min_mem = min(r[1] for r in self.records)

        print(f"\n{'=' * 60}")
        print("GPU 显存监控报告")
        print(f"{'=' * 60}")
        print(f"最小显存: {min_mem:.1f} MB")
        print(f"最大显存: {max_mem:.1f} MB")
        print(f"显存增量: {max_mem - min_mem:.1f} MB")

        # 打印标记点
        marks = [r for r in self.records if len(r) == 3]
        if marks:
            print("\n标记点:")
            for ts, mem, label in marks:
                print(f"  {label}: {mem:.1f} MB")


# 全局实例
_monitor = None


def start_monitor(gpu_id=0, interval=0.3):
    global _monitor
    _monitor = GPUMemoryMonitor(gpu_id, interval)
    _monitor.start()
    return _monitor


def mark(label):
    if _monitor:
        _monitor.mark(label)


def stop_and_report():
    if _monitor:
        _monitor.stop()
        _monitor.report()


if __name__ == "__main__":
    # 测试
    monitor = start_monitor()

    print("\n>>> 开始加载 vLLM...")
    mark("before_import")

    from vllm import LLM, SamplingParams

    mark("after_import")

    print("\n>>> 创建 LLM 实例...")
    try:
        llm = LLM(
            model="openai/gpt-oss-20b",  # 修改路径
            gpu_memory_utilization=0.8,
            max_model_len=128,
            enforce_eager=True,
        )
        mark("after_model_load")

        print("\n>>> 运行推理...")
        output = llm.generate(["Hello"], SamplingParams(max_tokens=1))
        mark("after_inference")

    except Exception as e:
        mark(f"error: {type(e).__name__}")
        print(f"Error: {e}")

    stop_and_report()
