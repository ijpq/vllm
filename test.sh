#!/bin/bash
set -e
# 设置运行次数
count=10

for ((i=1; i<=count; i++))
do
    echo "Starting run #$i..."
    python3 -m pytest tests/kernels/moe/test_fused_routing.py -vv -s && \
    python3 -m pytest tests/kernels/moe/test_gpt_oss_triton_kernels.py -vv -s
    
    echo "Finished run #$i"
    echo "-----------------------------------"
done