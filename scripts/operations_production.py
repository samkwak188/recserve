#!/usr/bin/env python3
"""Build once, then test exactly those immutable images; no publication/deployment."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run([sys.executable, 'scripts/build_production.py'], cwd=ROOT,
                            capture_output=True, text=True, check=True, timeout=1800)
    print(result.stdout, end='', flush=True)
    report = Path(result.stdout.strip().splitlines()[-1]).resolve()
    if not report.is_relative_to(ROOT / '.cache/production/images') or not report.is_file():
        raise RuntimeError('Build did not return an owned immutable image report')
    for script in ('check_operations.py', 'check_recovery.py'):
        subprocess.run([sys.executable, 'scripts/' + script, '--images', str(report)], cwd=ROOT,
                       check=True, timeout=600)


if __name__ == '__main__':
    main()
