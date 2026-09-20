#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .cache/venv
export IMPLICIT_DISABLE_CUDA=1  # ALS data preparation uses the CPU implementation.
python3 scripts/run_logged.py --name python-dependencies --timeout 1200 -- .cache/venv/bin/python -m pip install -r requirements.txt
