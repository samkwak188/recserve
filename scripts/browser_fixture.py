#!/usr/bin/env python3
"""Ephemeral synthetic accounts + HTTPS API + real C++ retrieval for Playwright."""
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from sqlalchemy import insert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recserve_app.db import Database
from recserve_app.schema_v1 import invitations
from scripts.model_v2 import write_bundle
from scripts.check_browser import web_command


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    base = ROOT / '.cache/production/browser' / str(time.time_ns())
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fixture-', dir=base) as temp:
        directory = Path(temp)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
        certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(hours=2))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'),
                x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False).sign(key, hashes.SHA256()))
        (directory / 'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        (directory / 'cert.pem').write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        movies = [dict(id=i + 1, title=f'Fixture film {i + 1}', year=1990 + i % 25, genres=['Drama'], available=True) for i in range(100)]
        manifest = write_bundle(directory / 'model', np.random.default_rng(7).normal(size=(100, 8)).astype(np.float32),
            movies, list(range(1, 101)), 'als', {'dataset': 'synthetic-browser-fixture'}, 'Synthetic test data', ROOT / 'build-production/recserve_fixture')
        ports = set()
        while len(ports) < 3:
            ports.add(free_port())
        api_port, retrieval_port, metrics_port = sorted(ports)
        origin = f'https://localhost:{api_port}'
        database = Database(os.environ['DATABASE_URL'])
        accounts = []
        with database.engine.begin() as tx:
            tx.execute(insert(invitations), [{'email': f'browser-{i}@example.invalid'} for i in range(2)])
        for i in range(2):
            token, csrf = database.login(f'browser-{i}', f'browser-{i}@example.invalid')
            accounts.append(dict(token=token, csrf=csrf))
        database.engine.dispose()
        fixture = directory / 'accounts.json'
        fixture.write_text(json.dumps(dict(origin=origin, accounts=accounts)))
        fixture.chmod(0o600)
        env = dict(os.environ, APP_ORIGIN=origin, MODEL_BUNDLE=str(manifest), GOOGLE_CLIENT_ID='browser-fixture',
            GOOGLE_CLIENT_SECRET='browser-fixture', RETRIEVAL_PORT=str(retrieval_port),
            PLAYWRIGHT_BASE_URL=origin, BROWSER_FIXTURE=str(fixture), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
            BROWSER_OUTPUT_DIR=str(base), PYTHONPATH=str(ROOT))
        processes = []
        with (base / 'services.log').open('w') as log:
            try:
                processes.append(subprocess.Popen([sys.executable, 'scripts/model_v2.py', 'serve', '--manifest', str(manifest),
                    '--port', str(retrieval_port), '--metrics-port', str(metrics_port)], cwd=ROOT, env=env, stdout=log, stderr=log))
                processes.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'browser_app:build', '--factory',
                    '--app-dir', 'tests/production', '--host', '127.0.0.1', '--port', str(api_port),
                    '--ssl-certfile', str(directory / 'cert.pem'), '--ssl-keyfile', str(directory / 'key.pem'),
                    '--no-access-log'], cwd=ROOT, env=env, stdout=log, stderr=log))
                context = ssl.create_default_context(cafile=str(directory / 'cert.pem'))
                for _ in range(200):
                    try:
                        with urllib.request.urlopen(origin + '/readyz', context=context, timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        if any(process.poll() is not None for process in processes):
                            raise RuntimeError('Browser fixture process exited; inspect services.log')
                        time.sleep(.05)
                else:
                    raise RuntimeError('Browser fixture readiness timeout')
                subprocess.run(web_command('test', origin, str(fixture), str(base)), cwd=ROOT / 'web', env=env, check=True)
            finally:
                for process in reversed(processes):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == '__main__':
    main()
