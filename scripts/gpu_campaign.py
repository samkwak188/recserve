#!/usr/bin/env python3
"""Reproducible throughput characterization; batch latency excludes queue fill."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    files = sorted(p for folder in ('apps', 'include', 'src') for p in (ROOT / folder).rglob('*') if p.is_file())
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), platform=platform.platform(),
                cpu=platform.processor(), logical_cpus=os.cpu_count(),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_sha256=hashlib.sha256(''.join(digest(p) for p in files).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[4096, 16384, 65536, 262144, 1000000])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 8, 32, 128, 512])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--build', default='build-gpu')
    parser.add_argument('--out', default='results/gpu-crossover.json')
    args = parser.parse_args()
    if args.repeats < 1 or args.iterations < 1 or any(n <= 0 for n in args.sizes):
        parser.error('positive sizes and repeats required')
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = ROOT / '.cache/logs/gpu-campaign'
    logs.mkdir(parents=True, exist_ok=True)
    binary = ROOT / args.build / 'recserve_gpu_bench'
    fixture = ROOT / args.build / 'recserve_fixture'
    payload = dict(protocol='preformed batches; CPU baseline single-threaded; queue fill and network excluded; p99 descriptive only',
                   config=vars(args), provenance=provenance(), fixtures={}, measurements=[])
    payload['provenance']['binary_sha256'] = digest(binary)
    for label, command in [('compiler', ['g++', '--version']), ('cuda', ['/usr/local/cuda-13.1/bin/nvcc', '--version']),
                           ('gpu', ['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,driver_version,compute_cap,memory.total,power.limit', '--format=csv,noheader'])]:
        result = subprocess.run(command, capture_output=True, text=True)
        payload['provenance'][label] = result.stdout.strip()

    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, indent=2)+'\n')
        temp.replace(output)

    for n in args.sizes:
        catalog = ROOT / f'data/gpu_catalog_{n}.bin'
        index = ROOT / f'data/gpu_index_{n}.bin'
        if not catalog.exists() or not index.exists():
            with (logs / f'fixture-{n}.log').open('w') as log:
                subprocess.run([str(fixture), '--items', str(n), '--dim', '64', '--clusters', str(min(4096, max(64, n // 64))),
                                '--build-threads', '1', '--out-catalog', str(catalog), '--out-index', str(index)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        payload['fixtures'][str(n)] = dict(catalog_sha256=digest(catalog), index_sha256=digest(index))
        points = [('simd', 1), ('int8', 1), ('hnsw', 1)] + [('cuda', b) for b in args.batches]
        for repeat in range(args.repeats):
            for backend, batch in (points if repeat % 2 == 0 else reversed(points)):
                command = [str(binary), '--catalog', str(catalog), '--index', str(index), '--backend', backend,
                           '--batch', str(batch), '--iterations', str(args.iterations), '--ef', '384']
                stamp = time.monotonic()
                result = subprocess.run(command, capture_output=True, text=True, timeout=600)
                (logs / f'{n}-{backend}-{batch}-{repeat}.log').write_text(result.stdout+result.stderr)
                result.check_returncode()
                row = json.loads(result.stdout.strip().splitlines()[-1])
                row.update(repeat=repeat, process_seconds=time.monotonic()-stamp)
                if backend in ('cuda', 'simd') and row['recall'] < .999:
                    raise RuntimeError(f'exact backend failed quality: {row}')
                payload['measurements'].append(row)
                save()
        print(f'completed items={n}: {len(points) * args.repeats} processes', flush=True)
    summary = []
    for n in args.sizes:
        rows = [r for r in payload['measurements'] if r['items'] == n]
        cpu = statistics.median(r['qps'] for r in rows if r['backend'] == 'simd')
        ann = statistics.median(r['qps'] for r in rows if r['backend'] == 'hnsw')
        for b in args.batches:
            gpu = [r for r in rows if r['backend'] == 'cuda' and r['batch'] == b]
            qps = statistics.median(r['qps'] for r in gpu)
            summary.append(dict(items=n, batch=b, gpu_qps=qps, cpu_exact_qps=cpu, cpu_hnsw_qps=ann,
                                gpu_vs_exact=qps/cpu, gpu_vs_hnsw=qps/ann,
                                gpu_p99_batch_us=statistics.median(r['p99_batch_us'] for r in gpu),
                                hnsw_recall_min=min(r['recall'] for r in rows if r['backend'] == 'hnsw')))
    payload['summary'] = summary
    save()
    print(f'completed campaign: {output}')


if __name__ == '__main__':
    main()
