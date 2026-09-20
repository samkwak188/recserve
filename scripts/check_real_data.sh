#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=.cache/venv/bin/python
python3 scripts/run_logged.py --name movielens-train --timeout 900 -- "$py" scripts/prep_movielens.py --dataset small --prefix wsl_small --factors 32 --iterations 15
python3 scripts/run_logged.py --name movielens-index -- build/recserve_fixture --in-catalog data/wsl_small_catalog.bin --out-index data/wsl_small_index.bin --m 16 --ef-construction 200 --build-threads 1
python3 scripts/run_logged.py --name movielens-quality -- "$py" scripts/eval_baselines.py --prefix wsl_small --k 10 --max-cpp-delta 0.002 --out results/wsl-quality.json
python3 scripts/run_logged.py --name bundle-create -- python3 scripts/bundle.py create --manifest data/wsl_small_bundle.json --meta data/wsl_small_meta.json --index data/wsl_small_index.bin --version wsl-movielens-small-v1
