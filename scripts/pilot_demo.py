"""Self-contained real-TCP feedback/restart acceptance demo, with a JSON receipt."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bundle import sha256
from recserve_pilot.model import Model
from recserve_pilot.store import now_ms


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def fixture(root, build):
    items = [(1-i/16, i/16) for i in range(12)]
    (root/'catalog.bin').write_bytes(struct.pack('<Iii', 0x43415431, 12, 2) +
                                   b''.join(struct.pack('<ff', *item) for item in items))
    (root/'queries.bin').write_bytes(struct.pack('<Iii2f', 0x51525931, 1, 2, 1, 0))
    (root/'users.csv').write_text('query_row,user_id\n0,demo-user\n')
    (root/'items.csv').write_text('item_row,item_id\n'+''.join(f'{i},movie-{i}\n' for i in range(12)))
    (root/'train.csv').write_text('query_row,item\n0,0\n')
    (root/'meta.json').write_text(json.dumps(dict(train_csv='train.csv')))
    binary = build/('recserve_fixture.exe' if os.name == 'nt' else 'recserve_fixture')
    subprocess.run([str(binary), '--in-catalog', str(root/'catalog.bin'), '--out-index', str(root/'index.bin'),
                    '--build-threads', '1'], check=True, stdout=subprocess.DEVNULL, timeout=30)
    bundle = dict(schema_version=1, model_version='deterministic-demo', catalog='catalog.bin',
                  queries='queries.bin', index='index.bin', user_map='users.csv', item_map='items.csv',
                  training_meta='meta.json', files={p.name: sha256(p) for p in root.iterdir() if p.is_file()})
    manifest = root/'bundle.json'
    manifest.write_text(json.dumps(bundle))
    return manifest


def call(port, path, data=None, expected=200, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                                     data=None if data is None else json.dumps(data).encode(), headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        result = json.loads(response.read())
        if response.status != expected:
            raise AssertionError((response.status, result, expected))
        return result


def ready(process, port, path):
    until = time.monotonic()+30
    while time.monotonic() < until and process.poll() is None:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=0.2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f'process failed readiness: exit={process.poll()}')


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run(build, manifest=None):
    build = Path(build).resolve()
    with tempfile.TemporaryDirectory(prefix='recserve-pilot-') as directory:
        root = Path(directory)
        manifest = Path(manifest).resolve() if manifest else fixture(root, build)
        model = Model(manifest)
        ports = set()
        while len(ports) < 3:
            ports.add(free_port())
        upstream, admin, api = ports
        binary = build/('recserve_server.exe' if os.name == 'nt' else 'recserve_server')
        server_command = [str(binary), '--catalog', str(model.root/model.bundle['catalog']),
                          '--queries', str(model.root/model.bundle['queries']), '--index', str(model.root/model.bundle['index']),
                          '--port', str(upstream), '--metrics-port', str(admin)]
        gateway_command = [sys.executable, '-m', 'recserve_pilot.service', '--manifest', str(manifest),
                           '--db', str(root/'feedback.sqlite'), '--port', str(api), '--upstream-port', str(upstream)]
        processes = []
        checks = []
        with tempfile.TemporaryFile(mode='w+') as log:
            def start(command, port, endpoint):
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                ready(process, port, endpoint)
                return process
            try:
                cpp = start(server_command, admin, '/readyz')
                gateway = start(gateway_command, api, '/healthz')
                user = next(iter(model.users))
                before = call(api, '/v1/recommend', dict(request_id='before', user_id=user, k=5))
                assert before['candidate_source'].startswith('cpp_retrieval') and len(before['items']) == 5
                assert not (set(x['item_id'] for x in before['items']) & model.seen[user])
                checks.append('real_cpp_retrieval_and_training_seen_exclusion')
                item = before['items'][0]['item_id']
                shown = dict(schema_version=1, event_id='shown', event_time_ms=now_ms(), user_id=user,
                             item_id=item, kind='shown', request_id='before')
                call(api, '/v1/events', shown)
                dismiss = dict(shown, event_id='dismiss', kind='dismiss')
                call(api, '/v1/events', dismiss)
                duplicate = call(api, '/v1/events', dismiss)
                assert duplicate['duplicate']
                call(api, '/v1/events', dict(dismiss, kind='save'), expected=409)
                checks.append('acknowledged_outcome_duplicate_and_conflict')
                after = call(api, '/v1/recommend', dict(request_id='after', user_id=user, k=5))
                assert item not in [x['item_id'] for x in after['items']]
                assert after['feature_generation'] == 2 and before['items'] != after['items']
                checks.append('feedback_changes_served_response')
                # Kill both real processes; restart against the same database and bundle.
                stop(gateway)
                stop(cpp)
                cpp = start(server_command, admin, '/readyz')
                gateway = start(gateway_command, api, '/healthz')
                restored = call(api, '/v1/recommend', dict(request_id='restored', user_id=user, k=5))
                assert restored['items'] == after['items'] and restored['feature_generation'] == 2
                assert call(api, '/v1/events', dismiss)['duplicate']
                checks.append('full_stack_restart_and_replay')
                assert call(api, '/v1/recommend', dict(request_id='before', user_id=user, k=5)) == before
                call(api, '/v1/recommend', dict(request_id='before', user_id=user, k=4), expected=409)
                call(api, '/v1/recommend', dict(request_id='bad', user_id='not-a-user', k=5), expected=400)
                call(api, '/v1/recommend', {}, expected=403, origin='https://untrusted.example')
                checks.append('idempotent_response_identity_validation_and_browser_origin_rejection')
                cold = call(api, '/v1/recommend', dict(request_id='cold', user_id='guest:local-demo', k=5))
                assert cold['candidate_source'] == 'popularity_cold_start'
                stop(cpp)
                fallback = call(api, '/v1/recommend', dict(request_id='fallback', user_id=user, k=5))
                assert fallback['degraded'] and item not in [x['item_id'] for x in fallback['items']]
                checks.append('cold_start_and_explicit_upstream_failure_fallback')
                return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
                            fixture='provided_bundle' if model.version != 'deterministic-demo' else 'synthetic_correctness',
                            model_version=model.version, manifest_sha256=model.fingerprint,
                            binary_sha256=sha256(binary), checks=checks, before=before, after=after,
                            restored=restored, fallback=fallback,
                            limitations=['local single-host reference', 'not latency or user-impact evidence',
                                         'issued responses require explicit shown acknowledgements'])
            except BaseException:
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                for process in reversed(processes):
                    stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', default='build')
    parser.add_argument('--manifest')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = run(args.build, args.manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, checks=result['checks'], model_version=result['model_version'])))


if __name__ == '__main__':
    main()
