#!/usr/bin/env python3
"""Export the reviewed baseline without changing the working tree or Git refs."""
import pathlib
import subprocess
import tarfile

root = pathlib.Path(__file__).resolve().parents[1]
target = root/'.cache/base-source'
if not (target/'CMakeLists.txt').exists():
    target.mkdir(parents=True, exist_ok=True)
    archive = root/'.cache/baseline.tar'
    subprocess.run(['git', 'archive', '--format=tar', '--output='+str(archive),
                    '86bbbc22462c70ff5fa2307dadb161183bf9a324'], cwd=root, check=True)
    with tarfile.open(archive) as source:
        source.extractall(target, filter='data')
