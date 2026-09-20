#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_logged.py --name baseline-configure -- cmake -S . -B build-baseline -G Ninja -DCMAKE_BUILD_TYPE=Release
python3 scripts/run_logged.py --name baseline-build -- cmake --build build-baseline -j 4
python3 scripts/run_logged.py --name baseline-tests -- ctest --test-dir build-baseline --output-on-failure
python3 scripts/run_logged.py --name baseline-bench -- build-baseline/recserve_bench --items 4096 --n 200 --trials 3 --json
