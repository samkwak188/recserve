#!/usr/bin/env python3
"""Build locally, record immutable image IDs, never publish or deploy."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.production_runner import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', choices=['app', 'web', 'database'], default=['app', 'web', 'database'])
    args = parser.parse_args()
    identity = source_identity(ROOT)
    directory = ROOT / '.cache/production/images' / str(time.time_ns())
    directory.mkdir(parents=True)

    def build(target):
        tag = 'recserve-pilot-' + target + ':' + identity['source_sha256'][:12]
        with (directory / (target + '.log')).open('w') as output:
            subprocess.run(['docker', 'build', '-f', 'Dockerfile.production', '--target', target,
                            '-t', tag, '.'], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
        result = json.loads(subprocess.check_output(['docker', 'image', 'inspect', tag]))[0]
        print(target + ': built ' + result['Id'], flush=True)
        return target, {'tag': tag, 'id': result['Id']}

    with ThreadPoolExecutor(max_workers=2) as pool:
        images = dict(pool.map(build, args.targets))
    report = dict(source=identity, images=images, completed_utc_ns=time.time_ns())
    (directory / 'images.json').write_text(json.dumps(report, indent=2) + '\n')
    print(directory / 'images.json')


if __name__ == '__main__':
    main()
