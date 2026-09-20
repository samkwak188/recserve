"""Real TCP integration: framing, invalid requests, slow clients, overload and drain."""
import argparse
from http.client import HTTPConnection
import json
import os
import pathlib
import socket
import struct
import subprocess
import tempfile
import time
import concurrent.futures
import threading

MAGIC = 0x52535631


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def read_exact(s, n):
    data = b''
    while len(data) < n:
        part = s.recv(n-len(data))
        if not part:
            raise EOFError('closed')
        data += part
    return data


def request(s, user=1, k=10, timeout=1000000):
    s.sendall(struct.pack('<IIQIIII', MAGIC, 24, 7, user, k, 32, timeout))
    magic, length = struct.unpack('<II', read_exact(s, 8))
    assert magic == MAGIC and 16 <= length <= 4112
    response = read_exact(s, length)
    rid, status, count = struct.unpack_from('<QII', response)
    assert rid == 7 and length == 16 + count * 8
    return status, [struct.unpack_from('<If', response, 16 + i*8) for i in range(count)]


def http(port, path):
    conn = HTTPConnection('127.0.0.1', port, timeout=2)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='build')
    parser.add_argument('--backend', default='hnsw')
    args = parser.parse_args()
    binary = pathlib.Path(args.build) / ('recserve_server.exe' if os.name == 'nt' else 'recserve_server')
    port, admin = free_port(), free_port()
    while admin == port:
        admin = free_port()
    with tempfile.TemporaryFile(mode='w+') as log:
        command = [str(binary.resolve()), '--synthetic', '--items', '128', '--dim', '17', '--workers', '2',
                   '--queue', '2', '--io-ms', '300', '--port', str(port), '--metrics-port', str(admin),
                   '--run-seconds', '5', '--backend', args.backend, '--batch-wait-us', '10000']
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        clients = []
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    assert http(admin, '/readyz') == (200, 'ok\n')
                    break
                except (OSError, AssertionError):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('server did not become ready')
                    time.sleep(.02)
            with socket.create_connection(('127.0.0.1', port), timeout=2) as s:
                status, items = request(s)
                assert status == 0 and len(items) == 10 and len({i[0] for i in items}) == 10
                assert request(s, k=0)[0] == 3
                assert request(s, user=0xFFFFFFFF)[0] == 3
                assert request(s, timeout=0)[0] == 3
                assert request(s)[0] == 0
            barrier = threading.Barrier(2)
            def parallel_request(_):
                with socket.create_connection(('127.0.0.1', port), timeout=2) as s:
                    barrier.wait(timeout=2)
                    return request(s)[0]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                assert list(pool.map(parallel_request, range(2))) == [0, 0]
            with socket.create_connection(('127.0.0.1', port), timeout=2) as s:
                s.sendall(struct.pack('<II', MAGIC, 0xFFFFFFFF))
                assert s.recv(1) == b''
            with socket.create_connection(('127.0.0.1', port), timeout=2) as s:
                s.sendall(b'R')
                stamp = time.monotonic()
                assert s.recv(1) == b''
                assert time.monotonic() - stamp < 1.5
            for _ in range(32):
                clients.append(socket.create_connection(('127.0.0.1', port), timeout=2))
            time.sleep(.05)
            code, raw = http(admin, '/metrics')
            metrics = {k: int(v) for k, v in (line.split() for line in raw.splitlines())}
            assert code == 200 and metrics['recserve_connections_rejected_total'] > 0
            assert metrics['recserve_connections_active'] <= 2 and metrics['recserve_connections_queued'] <= 2
            if args.backend == 'cuda':
                assert metrics['recserve_gpu_batch_max'] >= 2 and metrics['recserve_gpu_fallback_total'] == 0
            for s in clients:
                s.close()
            clients.clear()
            time.sleep(.4)
            with socket.create_connection(('127.0.0.1', port), timeout=2) as s:
                assert request(s)[0] == 0
            stamp = time.monotonic()
            if os.name != 'nt':
                process.terminate()
            assert process.wait(timeout=6) == 0
            if os.name != 'nt':
                assert time.monotonic() - stamp < 2
            print(json.dumps(dict(backend=args.backend, framing=True, invalid_requests=True,
                                 stalled_client=True, bounded_overload=True, recovery=True, graceful_shutdown=True,
                                 metrics=metrics)))
        finally:
            for s in clients:
                s.close()
            if process.poll() is None:
                process.kill()
                process.wait()
            log.seek(0)
            print(log.read())


if __name__ == '__main__':
    main()
