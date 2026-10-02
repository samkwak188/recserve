#!/usr/bin/env python3
"""Bounded, interleaved first-call diagnostics; profiler timings are not capacity results."""
import argparse
import datetime
import json
from pathlib import Path
import random
import socket
import sqlite3
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bundle
from scripts.pilot_demo import free_port, ready, stop
from scripts.production_runner import source_identity


def timeline(path):
    """Require actual CUDA activity and retain host calls separately from kernel time."""
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        runtime = [dict(r) for r in db.execute(
            'SELECT a.start,a.end,a.globalTid,a.correlationId,s.value AS name '
            'FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId ORDER BY a.start')]
        kernels = [dict(r) for r in db.execute(
            'SELECT a.start,a.end,a.correlationId,s.value AS name '
            'FROM CUPTI_ACTIVITY_KIND_KERNEL a JOIN StringIds s ON s.id=a.demangledName ORDER BY a.start')]
        ranges = [dict(r) for r in db.execute(
            'SELECT a.start,a.end,a.globalTid,coalesce(a.text,s.value) AS name FROM NVTX_EVENTS a '
            'LEFT JOIN StringIds s ON s.id=a.textId WHERE a.end IS NOT NULL ORDER BY a.start')]
    if not runtime or not kernels:
        raise RuntimeError('Profiler did not record CUDA runtime and device kernels')
    episodes = []
    current = None
    for call in runtime:
        if current is None and call['name'].startswith('cudaMemcpyAsync'):
            current = dict(start=call['start'], thread=call['globalTid'], calls=[])
        if current is not None and call['globalTid'] == current['thread']:
            current['calls'].append(call)
            if call['name'].startswith('cudaStreamSynchronize'):
                begin, end = current['start'], call['end']
                correlations = {c['correlationId'] for c in current['calls']}
                device = [k for k in kernels if k['correlationId'] in correlations]
                if any(k['start'] < begin or k['end'] > end for k in device):
                    raise RuntimeError('Correlated kernel lies outside the synchronized request interval')
                host_ranges = [r for r in ranges if r['globalTid'] == current['thread']
                               and begin <= r['start'] and r['end'] <= end]
                episodes.append(dict(
                    first_copy_to_sync_ms=(end-begin)/1e6,
                    runtime_calls=[dict(name=c['name'], offset_ms=(c['start']-begin)/1e6,
                        duration_ms=(c['end']-c['start'])/1e6, correlation_id=c['correlationId'])
                        for c in current['calls']],
                    library_ranges=[dict(name=r['name'], offset_ms=(r['start']-begin)/1e6,
                        duration_ms=(r['end']-r['start'])/1e6) for r in host_ranges],
                    kernels=[dict(name=k['name'], offset_ms=(k['start']-begin)/1e6,
                        duration_ms=(k['end']-k['start'])/1e6, correlation_id=k['correlationId'])
                        for k in device],
                    kernel_duration_sum_ms=sum(k['end']-k['start'] for k in device)/1e6))
                current = None
    if len(episodes) != 3 or any(not e['kernels'] for e in episodes):
        raise RuntimeError('Expected exactly three complete GPU request episodes with device kernels')
    return dict(runtime_events=len(runtime), kernel_events=len(kernels), episodes=episodes)


def workload(args):
    from scripts.service_campaign import metrics
    manifest = bundle.verify(args.manifest)
    base = args.manifest.resolve().parent
    port, admin = free_port(), free_port()
    while admin == port:
        admin = free_port()
    command = [str(args.binary.resolve()), '--catalog', str(base/manifest['catalog']),
        '--queries', str(base/manifest['queries']), '--index', str(base/manifest['index']),
        '--backend', 'cuda', '--batch', '1', '--batch-wait-us', '0',
        '--port', str(port), '--metrics-port', str(admin)]
    result = dict(command=command, requests=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            ready(process, admin, '/readyz')
            for number, pause in enumerate((0, .05, args.idle_seconds)):
                time.sleep(pause)
                start = time.monotonic_ns()
                with socket.create_connection(('127.0.0.1', port), timeout=3) as connection:
                    connection.settimeout(3)
                    connection.sendall(struct.pack('<IIQIIII', 0x52535631, 24, number, 0, 10, 64, 1000000))
                    def read(size):
                        data = b''
                        while len(data) < size:
                            part = connection.recv(size-len(data))
                            if not part:
                                raise RuntimeError('incomplete response')
                            data += part
                        return data
                    magic, size = struct.unpack('<II', read(8))
                    if magic != 0x52535631 or not 16 <= size <= 4112:
                        raise RuntimeError('invalid response frame')
                    payload = read(size)
                    rid, status, count = struct.unpack_from('<QII', payload)
                    if rid != number or status != 0 or count != 10 or size != 16 + 8*count:
                        raise RuntimeError('GPU request failed')
                    ids = [struct.unpack_from('<If', payload, 16+8*i)[0] for i in range(count)]
                result['requests'].append(dict(number=number, idle_before_s=pause,
                    client_ms=(time.monotonic_ns()-start)/1e6, item_ids=ids))
            observed = metrics(admin)
            result['gpu_metrics'] = {key: value for key, value in observed.items()
                if key.startswith('recserve_gpu_') and '_bucket' not in key}
            assert observed['recserve_gpu_queries_total'] == 3
            assert observed['recserve_gpu_fallback_total'] == 0
            assert all(r['item_ids'] == result['requests'][0]['item_ids'] for r in result['requests'])
            result['status'] = 'passed'
        finally:
            stop(process)
            args.output.write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsys', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('build-gpu/recserve_server'))
    parser.add_argument('--manifest', type=Path, default=Path('data/wsl_small_bundle.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--idle-seconds', type=float, default=2)
    parser.add_argument('--output', type=Path, default=Path('.cache/nsight/campaign/report.json'))
    parser.add_argument('--workload', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 0 <= args.idle_seconds <= 15:
        parser.error('experiment exceeds bounded limits')
    if args.workload:
        workload(args)
        return
    if not args.nsys or not args.nsys.is_file() or ' ' in str(args.nsys.resolve()):
        parser.error('provide an installed Nsight executable whose resolved path has no spaces')
    if args.output.exists():
        parser.error('output already exists; preserve previous evidence and choose a new directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = source_identity(ROOT)
    result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=identity, binary_sha256=bundle.sha256(args.binary),
        manifest_sha256=bundle.sha256(args.manifest),
        script_sha256=bundle.sha256(Path(__file__)),
        nsys_version=subprocess.check_output([str(args.nsys), '--version'], text=True).strip(),
        device=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
            '--format=csv,noheader'], text=True).strip(),
        scope='RSV1 single-query CUDA diagnostic; not the CPU-only RSV2 pilot, capacity, or sanitizer qualification.',
        design='Three sequential requests per fresh process: cold, 50ms gap, configured idle gap. Interleaved trace/control.',
        caveats=['CUDA tracing changes timings; controls have no profiler.',
            'Device kernel sums, host APIs and library ranges overlap and must not be added.',
            'CPU sampling and context-switch tracing disabled; no OS/debugger policy changes.',
            'Three requests do not establish tail latency or behavior after long idle periods.'],
        rows=[])
    order = [(repeat, traced) for repeat in range(args.repeats) for traced in (False, True)]
    random.Random(17).shuffle(order)
    try:
        for repeat, traced in order:
            stem = args.output.parent / (f'repeat-{repeat}-' + ('trace' if traced else 'control'))
            output = stem.with_suffix('.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--workload',
                '--binary', str(args.binary.resolve()), '--manifest', str(args.manifest.resolve()),
                '--idle-seconds', str(args.idle_seconds), '--output', str(output.resolve())]
            if traced:
                command = [str(args.nsys), 'profile', '--trace=cuda,cublas', '--sample=none',
                    '--cpuctxsw=none', '--stats=true', '--output='+str(stem.resolve()), *command]
            with stem.with_suffix('.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
            row = dict(repeat=repeat, traced=traced, command=command, workload=json.loads(output.read_text()))
            if traced:
                row['timeline'] = timeline(stem.with_suffix('.sqlite'))
                row['trace_sha256'] = bundle.sha256(stem.with_suffix('.nsys-rep'))
            result['rows'].append(row)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(repeat=repeat, traced=traced,
                client_ms=[r['client_ms'] for r in row['workload']['requests']])), flush=True)
        if source_identity(ROOT) != identity:
            raise RuntimeError('Source changed during diagnostic campaign')
        result['status'] = 'passed'
    finally:
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
