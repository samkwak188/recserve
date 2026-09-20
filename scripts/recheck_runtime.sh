#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for target in build build-gpu build-check-thread; do
  python3 scripts/run_logged.py --name "recheck-build-$target" -- cmake --build "$target" -j 4
  if [[ "$target" == build-check-thread ]]; then
    python3 scripts/run_logged.py --name "recheck-test-$target" --timeout 300 -- setarch "$(uname -m)" -R ctest --test-dir "$target" --output-on-failure
  else
    python3 scripts/run_logged.py --name "recheck-test-$target" --timeout 300 -- ctest --test-dir "$target" --output-on-failure
  fi
done
python3 scripts/run_logged.py --name python-unit -- python3 -m unittest discover -s tests -p 'test_*.py' -v
