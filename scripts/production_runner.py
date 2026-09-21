#!/usr/bin/env python3
"""Source-bound, resumable checks. Full logs stay in .cache/production/runs."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity(root: Path) -> dict:
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root
    ).decode().split('\0')
    files = {name: digest(root / name) if (root / name).is_file() else 'deleted'
             for name in sorted(set(names)) if name}
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
            'source_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def stop_tree(process: subprocess.Popen) -> None:
    if os.name == 'nt':
        # Only the child PID allocated by this runner and its descendants.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def execute(command: list[str], log: Path, timeout: int, root: Path = ROOT) -> int:
    with log.open('w', encoding='utf-8') as output:
        try:
            child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=os.name != 'nt')
        except OSError as exc:
            output.write(f'{type(exc).__name__}: command could not start\n')
            return 127
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_tree(child)
            output.write('\nRunner timeout; owned process tree terminated.\n')
            return 124
        except BaseException:
            stop_tree(child)
            raise


def reusable(receipt: dict, fingerprint: str, directory: Path) -> bool:
    if receipt.get('fingerprint') != fingerprint or receipt.get('status') != 'passed':
        return False
    steps = receipt.get('steps', [])
    logs_ok = bool(steps) and all(
        s.get('exit_code') == 0 and (directory / s['log']).is_file()
        and digest(directory / s['log']) == s.get('log_sha256') for s in steps)
    artifacts = receipt.get('artifacts', {})
    return logs_ok and all((ROOT / name).is_file() and digest(ROOT / name) == value
                           for name, value in artifacts.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'scripts/production_stages.json').read_text())
    if args.stage not in config:
        parser.error('unknown stage')
    stage = config[args.stage]
    identity = source_identity(ROOT)
    dependencies = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    identity.update(python=sys.version, platform=platform.platform(), dependencies=dependencies,
                    executable=str(Path(sys.executable).resolve()), stage=stage)
    for tool in ('cmake', 'c++', 'node', 'node.exe', 'docker'):
        try:
            identity[tool] = subprocess.check_output([tool, '--version'], stderr=subprocess.STDOUT,
                                                    text=True, timeout=10).splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            identity[tool] = 'unavailable'
    identity['inputs'] = {name: digest(ROOT / name) if (ROOT / name).is_file() else 'missing'
                          for name in stage.get('inputs', [])}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    runs = ROOT / '.cache/production/runs'
    runs.mkdir(parents=True, exist_ok=True)
    if args.resume:
        for path in sorted(runs.glob('*/receipt.json'), reverse=True):
            try:
                old = json.loads(path.read_text())
                if reusable(old, fingerprint, path.parent):
                    print(json.dumps({'status': 'reused', 'receipt': str(path)}))
                    return 0
            except (ValueError, KeyError, OSError):
                continue
    now = dt.datetime.now(dt.timezone.utc)
    directory = runs / (now.strftime('%Y%m%dT%H%M%SZ') + '-' + args.stage + '-' + uuid.uuid4().hex[:8])
    directory.mkdir()
    receipt = dict(stage=args.stage, fingerprint=fingerprint, identity=identity,
                   started_utc=now.isoformat(), status='running', steps=[])
    path = directory / 'receipt.json'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)

    save()
    code = 0
    try:
        for index, step in enumerate(stage['commands']):
            command = [sys.executable if arg == '{python}' else arg for arg in step['argv']]
            log = directory / f'{index:02d}-{step["name"]}.log'
            started = time.monotonic()
            code = execute(command, log, step.get('timeout_s', 300))
            receipt['steps'].append(dict(command=command, exit_code=code,
                elapsed_s=round(time.monotonic() - started, 3), log=log.name, log_sha256=digest(log)))
            save()
            print(f'{step["name"]}: {"PASS" if code == 0 else "FAIL"} ({code}); {log}', flush=True)
            if code:
                break
        if source_identity(ROOT) != {k: identity[k] for k in ('commit', 'source_sha256')}:
            receipt['reason'] = 'source changed during execution; evidence is not reusable'
            code = 3
        receipt['artifacts'] = {}
        for pattern in stage.get('artifacts', []):
            found = sorted(ROOT.glob(pattern))
            if not found:
                receipt['reason'] = 'required artifact missing: ' + pattern
                code = 3
            for artifact in found:
                if artifact.is_file():
                    receipt['artifacts'][artifact.relative_to(ROOT).as_posix()] = digest(artifact)
        receipt['status'] = 'passed' if code == 0 else 'failed'
    except BaseException:
        receipt['status'] = 'interrupted'
        raise
    finally:
        receipt['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps({'status': receipt['status'], 'receipt': str(path)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
