"""Interleaved cold/warm low-rate service experiment, not an automatic router."""
import argparse
import asyncio
import datetime
import json
import math
from pathlib import Path
import random
import subprocess
import time

import bundle
import open_loop
from pilot_demo import free_port, ready, stop
from service_campaign import metrics


def histogram(values, prefix):
    count = values.get(prefix+'_count', 0)
    boundaries = [1 << bit for bit in range(18)]
    buckets = [values.get(f'{prefix}_bucket{{le="{bound}"}}', 0) for bound in boundaries]
    upper = next((bound for bound, cumulative in zip(boundaries, buckets) if cumulative >= math.ceil(count*.99)), None) if count else None
    return dict(count=count, mean_us=values.get(prefix+'_sum', 0)/count if count else None,
                p99_bucket_upper_us=upper, cumulative_buckets=buckets,
                overflow=count-(buckets[-1] if buckets else 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default='build-gpu')
    parser.add_argument('--manifest', type=Path, default=Path('data/wsl_small_bundle.json'))
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rate', type=int, default=100)
    parser.add_argument('--out', type=Path, default=Path('results/gpu-lowrate-investigation.json'))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.seconds <= 60 or not 1 <= args.rate <= 5000:
        parser.error('invalid bounded experiment size')
    manifest = bundle.verify(args.manifest)
    root = args.manifest.resolve().parent
    binary = (Path(args.build)/'recserve_server').resolve()
    with (root/manifest['user_map']).open() as source:
        users = sum(1 for _ in source)-1
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  binary_sha256=bundle.sha256(binary), manifest_sha256=bundle.sha256(args.manifest),
                  source_sha256={name: bundle.sha256(Path(name)) for name in [
                      'apps/server.cpp', 'include/recserve/gpu_batcher.hpp', 'include/recserve/histogram.hpp',
                      'src/gpu_scorer.cu', 'scripts/investigate_gpu.py', 'scripts/open_loop.py']},
                  design='interleaved independent processes; 3 policies x cold/warm; fixed seed 17; one connection per arrival',
                  caveats=['same WSL desktop, not isolated hardware', 'warmup is 2 seconds at 1000 QPS',
                           'GPU event timing instrumentation is enabled for every batch',
                           'histogram percentiles are upper bucket bounds, not exact p99',
                           'GPU stages are per batch; scheduler and service stages are per request/connection',
                           'no pilot HTTP/database overhead included; not a trading benchmark'], rows=[])
    stages = ['connection_queue', 'request_compute', 'gpu_queue', 'gpu_wall', 'gpu_h2d', 'gpu_device_compute', 'gpu_d2h']
    settings = [(backend, batch, wait, warm) for backend, batch, wait in
                [('hnsw', 1, 0), ('cuda', 1, 0), ('cuda', 8, 100)] for warm in [False, True]]
    rng = random.Random(17)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logs = Path('.cache/logs'); logs.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        order = settings.copy(); rng.shuffle(order)
        for backend, batch, wait, warm in order:
            port, admin = free_port(), free_port()
            while admin == port:
                admin = free_port()
            name = f'gpu-lowrate-{repeat}-{backend}-b{batch}-warm{int(warm)}'
            with (logs/(name+'.log')).open('w') as log:
                process = subprocess.Popen([str(binary), '--catalog', str(root/manifest['catalog']),
                    '--queries', str(root/manifest['queries']), '--index', str(root/manifest['index']),
                    '--backend', backend, '--batch', str(batch), '--batch-wait-us', str(wait),
                    '--port', str(port), '--metrics-port', str(admin)], stdout=log, stderr=subprocess.STDOUT)
                try:
                    ready(process, admin, '/readyz')
                    load = argparse.Namespace(host='127.0.0.1', port=port, rate=args.rate, seconds=args.seconds,
                        concurrency=32, queue=256, deadline_ms=5, users=users, k=10, retrieve_k=64)
                    warm_result = None
                    if warm:
                        warm_result = asyncio.run(open_loop.campaign(argparse.Namespace(**dict(vars(load), rate=1000, seconds=2))))
                    time.sleep(.1)
                    before = metrics(admin)
                    measured = asyncio.run(open_loop.campaign(load))
                    time.sleep(.1)
                    after = metrics(admin)
                    delta = {key: value-before.get(key, 0) for key, value in after.items()}
                    row = dict(repeat=repeat, backend=backend, batch=batch, wait_us=wait, warmed=warm,
                               client=measured, warmup_failures=None if warm_result is None else warm_result['failure_rate'],
                               stages={stage: histogram(delta, 'recserve_'+stage+'_microseconds') for stage in stages},
                               gpu_batches=delta.get('recserve_gpu_batches_total', 0),
                               gpu_queries=delta.get('recserve_gpu_queries_total', 0),
                               gpu_expired=delta.get('recserve_gpu_expired_total', 0),
                               first_batch_wall_us=after.get('recserve_gpu_first_batch_wall_microseconds'),
                               first_batch_device_us=after.get('recserve_gpu_first_batch_device_microseconds'),
                               gpu_fallback=delta.get('recserve_gpu_fallback_total', 0))
                    result['rows'].append(row)
                    args.out.write_text(json.dumps(result, indent=2)+'\n')
                    print(f'{name} failures={measured["failure_rate"]:.4f} p99_us={measured["success_p99_us"]}', flush=True)
                finally:
                    stop(process)


if __name__ == '__main__':
    main()
