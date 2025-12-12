# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projec
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity, record_function
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
import os
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "True"
os.environ["VLLM_TORCH_CUDA_PROFILE"] = "True"

def main():
    # Create an LLM.
    llm = LLM(model="openai/gpt-oss-20b", enforce_eager=True)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    llm.start_profile()  # 开始profiling
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
    # with profile(
    # activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True
    # ) as prof:
    #     main()
    # prof.export_chrome_trace("trace.json")
