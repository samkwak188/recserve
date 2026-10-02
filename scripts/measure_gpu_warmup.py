#!/usr/bin/env python3
"""Interleave explicit GPU readiness warmup with cold-start controls; retain every failure."""
import argparse
import asyncio
import datetime
import json
from pathlib import Path
import random
import subprocess
import time

import bundle
import open_loop
from pilot_demo import free_port, ready, stop
from production_runner import source_identity
from service_campaign import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=Path('build-gpu'))
    parser.add_argument('--manifest', type=Path, default=Path('data/wsl_small_bundle.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--idle-seconds', type=float, default=15)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 0 <= args.idle_seconds <= 60 or not 1 <= args.seconds <= 10:
        parser.error('invalid bounded experiment parameters')
    if args.out.exists():
        parser.error('choose a new output path; previous measurements are never overwritten')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = bundle.verify(args.manifest)
    base = args.manifest.resolve().parent
    with (base / manifest['user_map']).open() as stream:
        users = sum(1 for _ in stream) - 1
    binary = (args.build / 'recserve_server').resolve()
    identity = source_identity(ROOT)
    report = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=identity, binary_sha256=bundle.sha256(binary), manifest_sha256=bundle.sha256(args.manifest),
        script_sha256=bundle.sha256(Path(__file__)),
        harness_sha256=bundle.sha256(Path(open_loop.__file__)),
        config=dict(repeats=args.repeats, idle_seconds=args.idle_seconds, seconds=args.seconds,
            rate=100, burst_rate=1000, deadline_ms=5, batch_limits=[1, 8], wait_us=100),
        device=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        scope='RSV1 CUDA startup policy experiment; no pilot HTTP/database overhead or production capacity claim.',
        design='Fixed-seed interleaved independent service processes. First request, low rate, idle, first after idle, burst.',
        caveats=['Shared desktop under WSL, not isolated hardware.',
            'Readiness uses 50ms polling; startup timing includes artifact and CUDA initialization.',
            'Five-millisecond deadlines include intended arrival, client queue and transport.',
            'Warmup does not count as serving batches or successful requests.',
            'No profiler attached. Idle is not a GPU reset. Failed requests remain in every stage.',
            'Small sample of restarts and short bursts; not a long soak or latency guarantee.'],
        rows=[])
    cases = [(repeat, batch, warmup) for repeat in range(args.repeats)
             for batch in (1, 8) for warmup in (False, True)]
    random.Random(31).shuffle(cases)
    try:
        for repeat, batch, warmup in cases:
            port, admin = free_port(), free_port()
            while port == admin:
                admin = free_port()
            name = f'repeat-{repeat}-batch-{batch}-warmup-{int(warmup)}'
            command = [str(binary), '--catalog', str(base/manifest['catalog']),
                '--queries', str(base/manifest['queries']), '--index', str(base/manifest['index']),
                '--backend', 'cuda', '--batch', str(batch), '--batch-wait-us', '100',
                '--port', str(port), '--metrics-port', str(admin)]
            if warmup:
                command.append('--gpu-warmup')
            row = dict(repeat=repeat, batch=batch, warmup=warmup, command=command, stages={})
            report['rows'].append(row)
            log_path = args.out.parent/(name + '.log')
            with log_path.open('w') as log:
                started = time.monotonic()
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    ready(process, admin, '/readyz')
                    row['startup_ms'] = (time.monotonic()-started)*1000
                    row['initial_metrics'] = metrics(admin)
                    initial = row['initial_metrics']
                    assert initial['recserve_gpu_warmup_batches_total'] == (batch if warmup else 0)
                    assert initial['recserve_gpu_queries_total'] == 0
                    assert initial['recserve_gpu_batches_total'] == 0
                    assert initial['recserve_gpu_first_batch_wall_microseconds'] == 0
                    config = dict(host='127.0.0.1', port=port, concurrency=32, queue=256,
                        deadline_ms=5, users=users, k=10, retrieve_k=64)
                    def load(stage, rate, seconds):
                        measured = asyncio.run(open_loop.campaign(argparse.Namespace(
                            **config, rate=rate, seconds=seconds)))
                        assert sum(measured['counts'].values()) == measured['attempted']
                        assert measured['success_p99_us'] is None or measured['success_p99_us'] < 5000
                        row['stages'][stage] = measured
                        args.out.write_text(json.dumps(report, indent=2)+'\n')
                        print(json.dumps(dict(case=name, stage=stage, counts=measured['counts'],
                            attempted=measured['attempted'])), flush=True)
                    load('first_request', 1, 1)
                    load('low_rate', 100, args.seconds)
                    time.sleep(args.idle_seconds)
                    load('first_after_idle', 1, 1)
                    load('burst', 1000, args.seconds)
                    time.sleep(.2) # Drain the bounded outstanding GPU work before reading counters.
                    row['final_metrics'] = metrics(admin)
                    assert row['final_metrics']['recserve_gpu_healthy'] == 1
                    assert row['final_metrics']['recserve_gpu_fallback_total'] == 0
                    row['status'] = 'completed'
                finally:
                    stop(process)
                    row['server_exit_code'] = process.returncode
                    row['log_sha256'] = bundle.sha256(log_path)
                    args.out.write_text(json.dumps(report, indent=2)+'\n')
            if process.returncode != 0:
                raise RuntimeError('server did not stop cleanly')
        if source_identity(ROOT) != identity:
            raise RuntimeError('source changed during campaign')
        report['status'] = 'completed'
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
