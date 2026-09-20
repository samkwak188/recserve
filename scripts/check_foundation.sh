#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_logged.py --name python-unit -- python3 -m unittest discover -s tests -p 'test_*.py' -v
for mode in none address undefined thread; do
  target="build-check-$mode"
  python3 scripts/run_logged.py --name "configure-$mode" -- cmake -S . -B "$target" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRECSERVE_SANITIZER="$mode"
  python3 scripts/run_logged.py --name "build-$mode" -- cmake --build "$target" -j 4
  if [[ "$mode" == thread && "${RECSERVE_TSAN_NO_ASLR:-0}" == 1 ]]; then
    python3 scripts/run_logged.py --name "test-$mode" --timeout 600 -- setarch "$(uname -m)" -R ctest --test-dir "$target" --output-on-failure
  else
    python3 scripts/run_logged.py --name "test-$mode" --timeout 600 -- ctest --test-dir "$target" --output-on-failure
  fi
done
