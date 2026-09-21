#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-all}" in all|tests-only) ;; *) echo 'usage: check_ownership_runtime.sh [all|tests-only]' >&2; exit 2 ;; esac
for target in build build-gpu build-check-address build-check-undefined build-check-thread; do
  python3 scripts/run_logged.py --name "ownership-build-$target" -- cmake --build "$target" -j 4
  if [[ "$target" == build-check-thread ]]; then
    python3 scripts/run_logged.py --name "ownership-test-$target" --timeout 300 -- setarch "$(uname -m)" -R ctest --test-dir "$target" --output-on-failure
  else
    python3 scripts/run_logged.py --name "ownership-test-$target" --timeout 300 -- ctest --test-dir "$target" --output-on-failure
  fi
done
if [[ "${1:-all}" == tests-only ]]; then exit 0; fi
python3 scripts/run_logged.py --name ownership-gpu-investigation --timeout 600 -- python3 scripts/investigate_gpu.py
python3 scripts/summarize_gpu_investigation.py
