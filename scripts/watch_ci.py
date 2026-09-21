"""Bounded read-only CI wait; save full status with ci_status.py, never infer green."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if not 30 <= args.timeout <= 3600 or not 7 <= len(args.sha) <= 40 or any(c not in '0123456789abcdef' for c in args.sha):
        parser.error('valid hex commit and bounded timeout required')
    root = Path(__file__).resolve().parents[1]
    deadline = time.monotonic()+args.timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run([sys.executable, str(root/'scripts/ci_status.py'), '--sha', args.sha],
                                    cwd=root, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            print('CI status read timed out; no conclusion inferred', flush=True)
        else:
            if result.returncode:
                print('CI status temporarily unavailable; no conclusion inferred', flush=True)
            else:
                status = json.loads(result.stdout)
                print(json.dumps(status), flush=True)
                runs = status['runs']
                if runs and all(run['status'] == 'completed' for run in runs):
                    return 0 if all(run['conclusion'] == 'success' for run in runs) else 1
        time.sleep(min(30, max(0, deadline-time.monotonic())))
    return 124


if __name__ == '__main__':
    raise SystemExit(main())
