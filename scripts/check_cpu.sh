#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_logged.py --name cpu-configure -- cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRECSERVE_WITH_AVX2=ON -DRECSERVE_WITH_RDKAFKA=ON
python3 scripts/run_logged.py --name cpu-build -- cmake --build build -j 4
python3 scripts/run_logged.py --name cpu-tests -- ctest --test-dir build --output-on-failure
python3 scripts/run_logged.py --name cpu-quick --timeout 1800 -- python3 scripts/measure.py --quick --out results/wsl-cpu.json --board results/WSL_CPU.md
