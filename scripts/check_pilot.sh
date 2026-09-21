#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_logged.py --name pilot-configure -- cmake -S . -B build
python3 scripts/run_logged.py --name pilot-build -- cmake --build build -j 4
python3 scripts/run_logged.py --name pilot-python --timeout 180 -- python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/run_logged.py --name pilot-ctest --timeout 300 -- ctest --test-dir build --output-on-failure
python3 scripts/run_logged.py --name pilot-demo --timeout 90 -- python3 scripts/pilot_demo.py --build build --out results/pilot-synthetic.json
if [[ -f data/wsl_small_bundle.json ]]; then
  python3 scripts/run_logged.py --name pilot-movielens --timeout 90 -- python3 scripts/pilot_demo.py --build build --manifest data/wsl_small_bundle.json --out results/pilot-movielens.json
fi
