#!/usr/bin/env python3
"""Resolve container base images to digests; only writes the requested lockfile."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

IMAGES = {'python': 'python:3.12-slim-bookworm', 'node': 'node:22-bookworm-slim',
          'caddy': 'caddy:2-alpine', 'postgres': 'postgres:17-bookworm'}


def resolve(entry):
    name, tag = entry
    subprocess.run(['docker', 'pull', tag], check=True)
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', tag]))[0]
    return name, dict(tag=tag, digest=info['RepoDigests'][0], image_id=info['Id'])


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        locked = dict(pool.map(resolve, IMAGES.items()))
    destination = Path('.cache/production/images.lock.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(locked, indent=2) + '\n')
    print(json.dumps(locked))
