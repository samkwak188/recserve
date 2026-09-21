#!/usr/bin/env python3
"""Run a real HTTPS browser fixture; the outer runner captures all output."""
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def web_command(stage, origin='', fixture='', output=''):
    if sys.platform == 'win32' or 'microsoft' in platform.release().lower():
        def windows(path):
            return str(path) if sys.platform == 'win32' else subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()
        command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            windows(ROOT / 'scripts/Check-Web.ps1'), '-Stage', stage]
        if origin:
            command += ['-BaseURL', origin]
        if fixture:
            command += ['-FixturePath', windows(Path(fixture))]
        if output:
            command += ['-OutputDirectory', windows(Path(output))]
        return command
    return ['npm', 'run', 'build'] if stage == 'build' else ['npm', 'test']


def main():
    subprocess.run([sys.executable, 'scripts/export_openapi.py', '--check'], cwd=ROOT, check=True)
    subprocess.run(web_command('build'), cwd=ROOT / 'web', check=True)
    subprocess.run([sys.executable, 'scripts/check_production.py', '--browser'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
