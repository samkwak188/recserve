#!/usr/bin/env python3
"""Build-independent CPU container smoke, isolation checks and clean teardown."""
import argparse
import json
import pathlib
import socket
import struct
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', default='docker')
    args = parser.parse_args()
    compose = [args.docker, 'compose', '-p', 'recserve-validation', '-f', 'compose.serve.yml']
    def run(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=120).strip()
    if run([*compose, 'ps', '-a', '-q']):
        raise RuntimeError('validation project already exists; refusing to change it')
    try:
        print(run([*compose, 'up', '-d', '--no-build']))
        cid = run([*compose, 'ps', '-q', 'recserve'])
        until = time.monotonic()+60
        while True:
            state = json.loads(run([args.docker, 'inspect', cid]))[0]
            if state['State'].get('Health', {}).get('Status') == 'healthy':
                break
            if not state['State']['Running'] or time.monotonic() > until:
                print(run([args.docker, 'logs', cid]))
                raise RuntimeError('container never became healthy')
            time.sleep(.5)
        def exact(s, n):
            data = b''
            while len(data) < n:
                part = s.recv(n-len(data))
                if not part: raise EOFError('container closed connection')
                data += part
            return data
        with socket.create_connection(('127.0.0.1', 9400), timeout=2) as s:
            s.sendall(struct.pack('<IIQIIII', 0x52535631, 24, 55, 1, 10, 64, 1000000))
            magic, length = struct.unpack('<II', exact(s, 8))
            assert magic == 0x52535631 and length == 96
            rid, status, count = struct.unpack_from('<QII', exact(s, length))
            assert (rid, status, count) == (55, 0, 10)
        assert state['Config']['User'] == '10001:10001'
        assert state['HostConfig']['ReadonlyRootfs'] and state['HostConfig']['PidsLimit'] == 128
        assert state['HostConfig']['PortBindings']['9400/tcp'][0]['HostIp'] == '127.0.0.1'
        result = dict(passed=True, image_id=state['Image'], user=state['Config']['User'],
                      readonly_root=True, loopback_only=True, pids_limit=128, health='healthy', real_model_request='ok')
        pathlib.Path('results/container-validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
    finally:
        print(run([*compose, 'down', '--timeout', '15']))


if __name__ == '__main__':
    main()
