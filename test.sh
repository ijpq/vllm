#!/bin/bash
set -e 

count=10

for ((i=1; i<=count; i++))
do
    echo "Starting run #$i..."
    
    python3 -m pytest tests/kernels/moe/test_fused_routing.py -vv -s || { echo "test_fused_routing failed"; exit 1; }
    
    python3 -m pytest tests/kernels/moe/test_gpt_oss_triton_kernels.py -vv -s || { echo "test_gpt_oss_triton_kernels failed"; exit 1; }

vllm bench latency \
    --model openai/gpt-oss-20b \
    --input-len 4096 \
    --output-len 1024 \
    --batch-size 32 \
    --num-iters-warmup 15 \
    --num-iters 10
# python3 examples/offline_inference/basic/basic.py|| { echo "with cuda graph failed"; exit 1; }
# python3 examples/offline_inference/basic/basic120b.py|| { echo "with cuda graph failed"; exit 1; }
    echo "Finished run #$i"
    echo "-----------------------------------"
done

echo "All $count runs passed successfully!"