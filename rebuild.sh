#!/bin/bash
# Quick rebuild script for fused_routing kernel
set -e

echo "Removing old .so files..."
rm -f ./vllm/*.so 2>/dev/null || true

echo "Rebuilding vllm..."
CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v 2>&1 | tail -50

echo "Build complete!"
