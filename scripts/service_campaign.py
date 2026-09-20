#!/usr/bin/env python3
"""Open-loop capacity sweep over a verified real-data bundle; no public listener."""
import argparse
import asyncio
import datetime
import json
import pathlib
import socket
import subprocess
import time
import urllib.request

import bundle
import open_loop


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def metrics(admin):
    with urllib.request.urlopen(f'http://127.0.0.1:{admin}/metrics', timeout=2) as result:
        return {k: int(v) for k, v in (line.split() for line in result.read().decode().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=pathlib.Path, default=pathlib.Path('data/wsl_small_bundle.json'))
    parser.add_argument('--build', default='build-gpu')
    parser.add_argument('--backends', nargs='+', default=['hnsw', 'cuda'])
    parser.add_argument('--rates', nargs='+', type=int, default=[100, 500, 1000, 2000])
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('results/service-capacity.json'))
    args = parser.parse_args()
    manifest = bundle.verify(args.manifest)
    data = args.manifest.resolve().parent
    binary = (pathlib.Path(args.build)/'recserve_server').resolve()
    with (data/manifest['user_map']).open() as source:
        users = sum(1 for _ in source)-1
    payload = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   binary_sha256=bundle.sha256(binary), manifest_sha256=bundle.sha256(args.manifest),
                   protocol='local TCP; 8 workers; client concurrency32; batch8 wait100us; latency from intended arrival',
                   provisional_slo=dict(p99_us=5000, failure_rate=.001), rows=[])
    logs = pathlib.Path('.cache/logs'); logs.mkdir(parents=True, exist_ok=True)
    for backend in args.backends:
        serving, admin = port(), port()
        while serving == admin:
            admin = port()
        with (logs/f'service-{backend}.log').open('w') as log:
            process = subprocess.Popen([str(binary), '--catalog', str(data/manifest['catalog']),
                '--index', str(data/manifest['index']), '--queries', str(data/manifest['queries']),
                '--backend', backend, '--port', str(serving), '--metrics-port', str(admin)], stdout=log, stderr=subprocess.STDOUT)
            try:
                until = time.monotonic()+30
                while True:
                    try:
                        metrics(admin); break
                    except OSError:
                        if process.poll() is not None or time.monotonic() > until:
                            raise RuntimeError('service startup failed')
                        time.sleep(.05)
                for rate in args.rates:
                    load = argparse.Namespace(host='127.0.0.1', port=serving, rate=rate, seconds=args.seconds,
                        concurrency=32, queue=256, deadline_ms=5, users=users, k=10, retrieve_k=64)
                    row = asyncio.run(open_loop.campaign(load))
                    row['backend'] = backend
                    row['metrics'] = metrics(admin)
                    row['provisional_slo_met'] = row['failure_rate'] <= .001 and row['success_p99_us'] is not None and row['success_p99_us'] <= 5000
                    payload['rows'].append(row)
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(payload, indent=2)+'\n')
                    print(f"{backend} offered={rate} ok={row['counts'].get('ok', 0)} failure={row['failure_rate']:.5f} p99_us={row['success_p99_us']}", flush=True)
                process.terminate()
                if process.wait(timeout=15) != 0:
                    raise RuntimeError('unclean service shutdown')
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()


if __name__ == '__main__':
    main()
