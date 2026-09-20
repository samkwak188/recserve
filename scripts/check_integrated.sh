#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python3 scripts/run_logged.py --name final-cpu-configure -- cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRECSERVE_WITH_AVX2=ON -DRECSERVE_WITH_RDKAFKA=ON
python3 scripts/run_logged.py --name final-cpu-build -- cmake --build build -j 4
python3 scripts/run_logged.py --name final-cpu-test -- ctest --test-dir build --output-on-failure
python3 scripts/run_logged.py --name final-gpu-test -- ctest --test-dir build-gpu --output-on-failure
python3 scripts/run_logged.py --name python-unit -- python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/run_logged.py --name hnsw-identical-queries -- .cache/venv/bin/python scripts/validate_hnsw.py --items 32768 --dim 64 --clusters 512 --queries 128 --out results/wsl-hnsw.json
python3 scripts/run_logged.py --name real-gpu-quality -- build-gpu/recserve_eval --catalog data/wsl_small_catalog.bin --queries data/wsl_small_queries.bin --test data/wsl_small_test.csv --train data/wsl_small_train.csv --kernel cuda --brute --json
