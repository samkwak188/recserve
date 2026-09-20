#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda-13.1/bin:$PATH
python3 scripts/run_logged.py --name gpu-configure -- cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DRECSERVE_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DRECSERVE_WITH_AVX2=ON
python3 scripts/run_logged.py --name gpu-build -- cmake --build build-gpu -j 4
python3 scripts/run_logged.py --name gpu-tests --timeout 600 -- ctest --test-dir build-gpu --output-on-failure
python3 scripts/run_logged.py --name gpu-memcheck --timeout 600 -- compute-sanitizer --tool memcheck --error-exitcode 1 build-gpu/recserve_gpu_tests
