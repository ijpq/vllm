# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

# 获取当前脚本的父目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的父目录（假设这是包的根目录）
parent_dir = os.path.dirname(current_dir)

# 将其加入系统路径
sys.path.append(parent_dir)

from gpu_monitor import mark, start_monitor, stop_and_report

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"


def main():
    # 启动显存监控
    monitor = start_monitor(interval=0.2)
    mark("00_initial")

    # 导入 vLLM (可能有 CUDA 初始化开销)
    from vllm import LLM, SamplingParams

    mark("01_after_import")

    # 创建采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 创建 LLM 实例 (模型加载)
    mark("02_before_llm_init")
    llm = LLM(
        model="openai/gpt-oss-20b",
        gpu_memory_utilization=0.8,
        max_model_len=128,
        enforce_eager=True,
        kv_cache_memory_bytes=500 * 1024 * 1024,  # 只分配 500 MB
        compilation_config=1,
    )
    mark("03_after_llm_init")

    # 第一次推理 (可能有额外初始化)
    mark("04_before_first_generate")
    outputs = llm.generate(prompts[:1], sampling_params)
    mark("05_after_first_generate")

    # 后续推理
    mark("06_before_batch_generate")
    outputs = llm.generate(prompts, sampling_params)
    mark("07_after_batch_generate")

    # 停止监控并打印报告
    stop_and_report()

    # 打印输出
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
