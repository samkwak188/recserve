#!/usr/bin/env python3
"""Run a command without a shell, retaining full output and a small JSON receipt."""
import argparse
import datetime
import json
import pathlib
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or pathlib.Path(args.name).name != args.name:
        parser.error('a command and a simple log name are required')
    root = pathlib.Path(__file__).resolve().parents[1]
    directory = root / '.cache' / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / (args.name + '.log')
    started = time.monotonic()
    receipt = dict(command=command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with log.open('w', encoding='utf-8') as output:
        try:
            process = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != 'nt')
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            if os.name != 'nt':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            code = 124
            output.write('\nTimed out\n')
        except OSError as exc:
            code = 127
            output.write(str(exc) + '\n')
    receipt.update(exit_code=code, elapsed_s=round(time.monotonic()-started, 2), log=str(log))
    log.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))
    if code:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]), file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
