#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python3 scripts/prepare_regression.py
python3 scripts/run_logged.py --name paired-base-configure -- cmake -S .cache/base-source -B build-regression-base -G Ninja -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS=-mavx2 -mfma'
python3 scripts/run_logged.py --name paired-base-build -- cmake --build build-regression-base -j 4
python3 scripts/run_logged.py --name paired-regression --timeout 900 -- python3 scripts/regression_gate.py --base-bench build-regression-base/recserve_bench --candidate-bench build/recserve_bench --fixture build/recserve_fixture --out results/wsl-regression.json
python3 scripts/run_logged.py --name final-cpu-quick --timeout 1800 -- python3 scripts/measure.py --quick --out results/wsl-cpu.json --board results/WSL_CPU.md
python3 scripts/run_logged.py --name gpu-campaign --timeout 7200 -- python3 scripts/gpu_campaign.py --iterations 200
python3 scripts/run_logged.py --name service-capacity --timeout 900 -- python3 scripts/service_campaign.py
