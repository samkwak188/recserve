#!/usr/bin/env python3
"""Create, verify, or serve an immutable catalog/index/query/identity bundle."""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import struct


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def member(root, name):
    path = (root / name).resolve()
    if pathlib.Path(name).name != name or path.parent != root.resolve():
        raise ValueError('bundle members must be files beside the manifest')
    return path


def verify(manifest):
    bundle = json.loads(manifest.read_text())
    if bundle.get('schema_version') != 1 or not bundle.get('model_version'):
        raise ValueError('unsupported bundle schema or missing model version')
    if not 1 <= len(bundle['files']) <= 32:
        raise ValueError('invalid member count')
    root = manifest.parent
    for name, expected in bundle['files'].items():
        if sha256(member(root, name)) != expected:
            raise ValueError('hash mismatch: ' + name)
    for role in ('catalog', 'queries', 'index', 'user_map', 'item_map'):
        if bundle[role] not in bundle['files']:
            raise ValueError('unverified bundle role: ' + role)
    def shape(role, magic):
        path = member(root, bundle[role])
        with path.open('rb') as source:
            code, count, dim = struct.unpack('<Iii', source.read(12))
        if code != magic or count <= 0 or not 1 <= dim <= 4096 or path.stat().st_size != 12 + count*dim*4:
            raise ValueError('invalid ' + role)
        return count, dim
    items, dim = shape('catalog', 0x43415431)
    users, query_dim = shape('queries', 0x51525931)
    if dim != query_dim:
        raise ValueError('catalog/query dimensions differ')
    with member(root, bundle['index']).open('rb') as source:
        magic, count, index_dim = struct.unpack('<Iii', source.read(12))
    if (magic, count, index_dim) != (0x48535732, items, dim):
        raise ValueError('index/catalog shapes differ')
    for role, expected, row_key, id_key in (('user_map', users, 'query_row', 'user_id'),
                                           ('item_map', items, 'item_row', 'item_id')):
        with member(root, bundle[role]).open(newline='') as source:
            rows = list(csv.DictReader(source))
        if len(rows) != expected or [int(r[row_key]) for r in rows] != list(range(expected)) or len({r[id_key] for r in rows}) != expected:
            raise ValueError('invalid identity map: ' + role)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'verify', 'serve'])
    parser.add_argument('--manifest', required=True, type=pathlib.Path)
    parser.add_argument('--meta', type=pathlib.Path)
    parser.add_argument('--index', type=pathlib.Path)
    parser.add_argument('--version', default='local-pilot')
    parser.add_argument('--binary', default='build/recserve_server')
    parser.add_argument('--backend', choices=['hnsw', 'simd', 'cuda'], default='hnsw')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9400)
    parser.add_argument('--metrics-port', type=int, default=9401)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    if args.action == 'create':
        if not args.meta or not args.index or args.meta.resolve().parent != root or args.index.resolve().parent != root:
            parser.error('metadata and index must be beside the manifest')
        meta = json.loads(args.meta.read_text())
        files = dict(meta['files'])
        files[args.meta.name] = sha256(args.meta)
        files[args.index.name] = sha256(args.index)
        prefix = meta['catalog'].removesuffix('_catalog.bin')
        bundle = dict(schema_version=1, model_version=args.version, files=files, catalog=meta['catalog'],
                      queries=meta['queries'], index=args.index.name, user_map=prefix+'_user_ids.csv',
                      item_map=prefix+'_item_ids.csv', training_meta=args.meta.name)
        temporary = args.manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(bundle, indent=2)+'\n')
        verify(temporary)
        temporary.replace(args.manifest)
    bundle = verify(args.manifest)
    print(json.dumps(dict(verified=True, manifest_sha256=sha256(args.manifest), model_version=bundle['model_version'])), flush=True)
    if args.action == 'serve':
        binary = str(pathlib.Path(args.binary).resolve())
        command = [binary, '--catalog', str(root/bundle['catalog']), '--queries', str(root/bundle['queries']),
                   '--index', str(root/bundle['index']), '--backend', args.backend, '--bind', args.bind,
                   '--port', str(args.port), '--metrics-port', str(args.metrics_port)]
        os.execv(binary, command)


if __name__ == '__main__':
    main()
