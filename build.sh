rm ./vllm/*.so && CMAKE_BUILD_TYPE=Debug CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v && python3 debug.py
