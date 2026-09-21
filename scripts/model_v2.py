#!/usr/bin/env python3
"""Build and launch verified item-only models. Outputs are immutable directories."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recserve_app.model import Model, file_hash


def write_bundle(directory, factors, movies, popularity, policy, training, license_text,
                 binary, regularization=.05):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    raw = np.asarray(factors, dtype=np.float32)
    normalized = (raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)).astype('<f4')
    with (directory / 'catalog.bin').open('wb') as stream:
        stream.write(struct.pack('<Iii', 0x43415431, *raw.shape))
        stream.write(normalized.tobytes())
    np.save(directory / 'factors.npy', raw, allow_pickle=False)
    raw64 = raw.astype(np.float64)
    np.save(directory / 'gram.npy', raw64.T @ raw64, allow_pickle=False)
    (directory / 'movies.json').write_text(json.dumps(movies, ensure_ascii=False) + '\n', encoding='utf-8')
    (directory / 'popularity.json').write_text(json.dumps(popularity) + '\n')
    (directory / 'LICENSE.txt').write_text(license_text, encoding='utf-8')
    subprocess.run([str(binary), '--in-catalog', str(directory / 'catalog.bin'), '--out-index',
        str(directory / 'index.bin'), '--m', '16', '--ef-construction', '200', '--build-threads', '1', '--queries', '1'], check=True)
    info = dict(schema_version=2, policy=policy, regularization=regularization,
                confidence={'like': 41, 'dislike': 11}, training=training,
                files={p.name: file_hash(p) for p in sorted(directory.iterdir())})
    manifest = directory / 'manifest.json'
    manifest.write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
    Model(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'serve'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--binary', default='build-production/recserve_server')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9400)
    parser.add_argument('--metrics-port', type=int, default=9401)
    args = parser.parse_args()
    model = Model(args.manifest)
    print(json.dumps({'verified': True, 'model_digest': model.digest}), flush=True)
    if args.action == 'serve':
        command = [args.binary, '--vectors-only', '--catalog', str(model.root / 'catalog.bin'),
            '--index', str(model.root / 'index.bin'), '--model-digest', model.digest,
            '--bind', args.bind, '--port', str(args.port), '--metrics-port', str(args.metrics_port),
            '--workers', '4', '--queue', '32', '--io-ms', '250']
        os.execv(str(Path(args.binary).resolve()), command)


if __name__ == '__main__':
    main()
