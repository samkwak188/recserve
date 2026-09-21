#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .cache/venv-production
.cache/venv-production/bin/python -m pip install --require-hashes -r requirements-production.lock
