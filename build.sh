rm ./vllm/*.so && CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v && python3 debug.py
