#!/usr/bin/env python3
"""Exercise immutable production images over Caddy HTTPS on an isolated local bridge."""
import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import secrets
import ssl
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
import numpy as np
from cryptography import x509
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.model_v2 import write_bundle
from scripts.production_runner import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    images = json.loads(args.images.read_text())['images']
    token = uuid.uuid4().hex[:12]
    prefix = 'recserve-stack-' + token
    directory = ROOT / '.cache/production/operations' / str(time.time_ns())
    directory.mkdir(parents=True)
    network, containers, volumes = prefix + '-network', [], []
    source = source_identity(ROOT)

    def run(argv, capture=False, **kwargs):
        return subprocess.run(argv, check=True, text=True, timeout=120,
            stdout=subprocess.PIPE if capture else None, **kwargs).stdout

    def wait(command):
        for _ in range(120):
            if subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0:
                return
            time.sleep(.5)
        raise RuntimeError('Container readiness timeout')

    def volume(role):
        name = prefix + '-' + role
        run(['docker', 'volume', 'create', '--label', 'recserve.proof=' + token, name])
        volumes.append(name)
        return name

    def start(role, target, extra, command=()):
        name = prefix + '-' + role
        containers.append(name)
        run(['docker', 'run', '-d', '--name', name, '--network', network, '--network-alias', role,
            '--label', 'recserve.proof=' + token, '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges:true', '--pids-limit', '96', '--memory', '512m',
            '--tmpfs', '/tmp:size=32m', *extra, images[target]['id'], *command])
        return name

    def bind(path, dest):
        return ['--mount', f'type=bind,source={path},target={dest},readonly']

    def envargs(values):
        return [part for key, value in values.items() for part in ('-e', key + '=' + str(value))]

    try:
        run(['docker', 'network', 'create', '--label', 'recserve.proof=' + token, network])
        with tempfile.TemporaryDirectory(prefix='fixture-', dir=directory) as temp:
            fixture = Path(temp)
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'ledger')])
            cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
                .not_valid_after(datetime.now(timezone.utc) + timedelta(hours=2))
                .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName([x509.DNSName('ledger'), x509.DNSName('localhost'),
                    x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False).sign(key, hashes.SHA256()))
            (fixture / 'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
            (fixture / 'cert.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            passwords = {name: secrets.token_hex(24) for name in ('owner', 'runtime')}
            files = dict(postgres_password=passwords['owner'],
                database_url=f"postgresql+psycopg://recserve_app:{passwords['runtime']}@postgres/recserve",
                migration_database_url=f"postgresql+psycopg://recserve_owner:{passwords['owner']}@postgres/recserve",
                google_client_secret='synthetic-test-only', privacy_access_key='synthetic-key',
                privacy_secret_key='synthetic-secret', privacy_fernet_key=Fernet.generate_key().decode())
            for filename, value in files.items():
                (fixture / filename).write_text(value)
            for path in fixture.iterdir():
                path.chmod(0o644)  # ephemeral files for isolated nonroot containers, never retained
            ledger = start('ledger', 'app', ['--entrypoint', 'python', *bind(fixture, '/fixture'),
                *bind(ROOT / 'tests/production/ledger_server.py', '/ledger_server.py')], ['/ledger_server.py'])
            data = volume('data')
            pg = start('postgres', 'database', ['--tmpfs', '/var/run/postgresql:uid=999,gid=999,mode=3775,size=8m',
                *bind(fixture / 'postgres_password', '/run/secrets/postgres_password'),
                '--mount', f'type=volume,source={data},target=/var/lib/postgresql/data',
                *envargs(dict(POSTGRES_PASSWORD_FILE='/run/secrets/postgres_password', POSTGRES_USER='recserve_owner', POSTGRES_DB='recserve'))])
            wait(['docker', 'exec', pg, 'pg_isready', '-U', 'recserve_owner', '-d', 'recserve'])
            migration = prefix + '-migration'
            containers.append(migration)
            run(['docker', 'run', '--name', migration, '--network', network, '--label', 'recserve.proof=' + token,
                '--read-only', '--cap-drop', 'ALL', '--tmpfs', '/tmp:size=32m',
                *bind(fixture / 'database_url', '/run/secrets/database_url'),
                *bind(fixture / 'migration_database_url', '/run/secrets/migration_database_url'),
                *envargs(dict(DATABASE_URL_FILE='/run/secrets/database_url', MIGRATION_DATABASE_URL_FILE='/run/secrets/migration_database_url')),
                images['app']['id'], 'migrate'])
            digests = {}
            for index, color in enumerate(('blue', 'green')):
                movies = [dict(id=i + 1, title=f'Operations movie {i + 1}', year=2000, genres=['Drama'], available=True) for i in range(100)]
                manifest = write_bundle(fixture / color, np.random.default_rng(7 + index).normal(size=(100, 8)).astype(np.float32),
                    movies, list(range(1, 101)), 'als', {'dataset': 'synthetic-operations-only'}, 'Synthetic fixture', ROOT / 'build-production/recserve_fixture')
                import hashlib
                digests[color] = hashlib.sha256(manifest.read_bytes()).hexdigest()
                start('retrieval-' + color, 'app', [*bind(manifest.parent, '/model'), '-e', 'MODEL_BUNDLE=/model/manifest.json'], ['retrieval'])
            routing = fixture / 'routing'
            routing.mkdir()
            upstream = routing / 'upstream.caddy'
            upstream.write_text('reverse_proxy api-blue:8000\n')
            web = start('web', 'web', ['-p', '127.0.0.1::8443', '-e', 'APP_HOSTNAME=localhost',
                *bind(routing, '/etc/recserve'), '--mount', f'type=volume,source={volume("caddy-data")},target=/data',
                '--mount', f'type=volume,source={volume("caddy-config")},target=/config'])
            port = json.loads(run(['docker', 'inspect', web], True))[0]['NetworkSettings']['Ports']['8443/tcp'][0]['HostPort']
            origin = 'https://localhost:' + port
            apis = {}
            for color in ('blue', 'green'):
                env = dict(APP_ORIGIN=origin, GOOGLE_CLIENT_ID='synthetic-test-only', MODEL_BUNDLE='/model/manifest.json',
                    DATABASE_URL_FILE='/run/secrets/database_url', GOOGLE_CLIENT_SECRET_FILE='/run/secrets/google_client_secret',
                    RETRIEVAL_HOST='retrieval-' + color, PRIVACY_S3_ENDPOINT='https://ledger:5000',
                    PRIVACY_S3_REGION='us-east-1', PRIVACY_S3_BUCKET='test-ledger', AWS_CA_BUNDLE='/fixture/cert.pem',
                    PRIVACY_S3_ACCESS_KEY_ID_FILE='/run/secrets/privacy_access_key',
                    PRIVACY_S3_SECRET_ACCESS_KEY_FILE='/run/secrets/privacy_secret_key',
                    PRIVACY_FERNET_KEY_FILE='/run/secrets/privacy_fernet_key')
                mounts = [*bind(fixture / color, '/model'), *bind(fixture / 'cert.pem', '/fixture/cert.pem'),
                    *bind(ROOT / 'tests/production/stack_seed.py', '/stack_seed.py')]
                for filename in ('database_url', 'google_client_secret', 'privacy_access_key', 'privacy_secret_key', 'privacy_fernet_key'):
                    mounts.extend(bind(fixture / filename, '/run/secrets/' + filename))
                apis[color] = start('api-' + color, 'app', [*mounts, *envargs(env)])
                wait(['docker', 'exec', apis[color], 'python', '-c',
                    "import urllib.request; urllib.request.urlopen('http://localhost:8000/readyz', timeout=2)"])
            # Caddy's internal CA is trusted only by this test client.
            certificate = fixture / 'caddy-root.crt'
            for _ in range(100):
                result = subprocess.run(['docker', 'cp', web + ':/data/caddy/pki/authorities/local/root.crt', str(certificate)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    break
                time.sleep(.1)
            context = ssl.create_default_context(cafile=str(certificate))
            accounts = json.loads(run(['docker', 'exec', apis['blue'], 'python', '/stack_seed.py'], True))
            # Failure assertions wait beyond Caddy's existing 10s write deadline.
            # This is not a latency qualification or a change to a release target.
            with httpx.Client(base_url=origin, verify=context, timeout=12, trust_env=False) as client:
                assert client.get('/').status_code == 200
                assert client.get('/internal/metrics').status_code == 404
                assert client.get('/api/v2/me').status_code == 401
                client.headers.update({'Cookie': '__Host-recserve=' + accounts[0]['token'],
                    'x-csrf-token': accounts[0]['csrf'], 'origin': origin})
                def post(path, payload):
                    response = client.post(path, json=payload)
                    assert response.status_code == 200, (path, response.status_code, response.text)
                    return response.json()
                assert client.put('/api/v2/me/consent', json={'adult': True, 'research': True}).status_code == 200
                pref = client.put('/api/v2/preferences', json={'request_id': str(uuid.uuid4()), 'changes': [{'item_id': 1, 'value': 1}]})
                assert pref.status_code == 200, pref.text
                request = {'request_id': str(uuid.uuid4()), 'k': 10}
                first = post('/api/v2/recommendations', request)
                assert first['model_version'] == digests['blue']
                assert all(item['id'] != 1 for item in first['items'])
                item_id = first['items'][0]['id']
                for kind in ('shown', 'save'):
                    post('/api/v2/events', {'event_id': str(uuid.uuid4()), 'request_id': request['request_id'], 'item_id': item_id, 'kind': kind, 'event_time_ms': int(time.time() * 1000)})
                def switch(color):
                    new = routing / 'next.caddy'
                    new.write_text('reverse_proxy api-' + color + ':8000\n')
                    os.replace(new, upstream)
                    run(['docker', 'exec', web, 'caddy', 'reload', '--config', '/etc/caddy/Caddyfile'])
                switch('green')
                fresh = post('/api/v2/recommendations', {'request_id': str(uuid.uuid4()), 'k': 10})
                assert fresh['model_version'] == digests['green']
                assert fresh['preference_revision'] == first['preference_revision'] + 1
                assert all(item['id'] not in (1, item_id) for item in fresh['items'])
                assert post('/api/v2/recommendations', request) == first
                rollback_start = time.monotonic()
                switch('blue')
                rolled = post('/api/v2/recommendations', {'request_id': str(uuid.uuid4()), 'k': 10})
                assert rolled['model_version'] == digests['blue']
                assert rolled['preference_revision'] == fresh['preference_revision']
                rollback_s = time.monotonic() - rollback_start
                run(['docker', 'stop', '-t', '2', prefix + '-retrieval-blue'])
                degraded = post('/api/v2/recommendations', {'request_id': str(uuid.uuid4()), 'k': 10})
                assert degraded['degraded'] and all(item['id'] not in (1, item_id) for item in degraded['items'])
                deletion = client.delete('/api/v2/me')
                assert deletion.status_code == 204, (deletion.status_code, deletion.text)
                assert client.get('/api/v2/me').status_code == 401
                client.headers.update({'Cookie': '__Host-recserve=' + accounts[1]['token'], 'x-csrf-token': accounts[1]['csrf']})
                run(['docker', 'stop', '-t', '2', ledger])
                assert client.delete('/api/v2/me').status_code == 503
                assert client.get('/api/v2/me').status_code == 200
                run(['docker', 'stop', '-t', '2', pg])
                assert client.get('/api/v2/me').status_code == 503
            environment = dict(os.environ, APP_HOSTNAME='localhost', GOOGLE_CLIENT_ID='synthetic-test-only',
                DATABASE_IMAGE=images['database']['id'], APP_IMAGE_BLUE=images['app']['id'], WEB_IMAGE=images['web']['id'],
                MODEL_DIR_BLUE=str(fixture / 'blue'), SECRETS_DIR=str(fixture), ROUTING_DIR=str(routing),
                PRIVACY_S3_ENDPOINT='https://ledger:5000', PRIVACY_S3_REGION='us-east-1', PRIVACY_S3_BUCKET='test-ledger', LOG_DRIVER='local')
            run(['docker', 'compose', '-f', 'compose.production.yml', 'config', '--quiet'], env=environment)
            report = dict(source=source, images=images, model_digests=digests, rollback_s=rollback_s,
                https=True, runtime_ddl_forbidden=True, cross_model_retries=True, preferences_survive_rollback=True,
                degraded_filtering=True, database_failure_closed=True, deletion_ledger_failure_closed=True,
                scope='local synthetic containers; fixture S3 transport; not cloud, OAuth or off-host qualification')
            (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'result': 'passed', 'rollback_s': rollback_s, 'report': str(directory / 'report.json')}))
    finally:
        for name in reversed(containers):
            result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
            if result.returncode == 0 and json.loads(result.stdout)[0]['Config'].get('Labels', {}).get('recserve.proof') == token:
                with (directory / (name + '.log')).open('w') as output:
                    subprocess.run(['docker', 'logs', name], stdout=output, stderr=subprocess.STDOUT, check=False)
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, check=False)
        for kind, names in (('volume', volumes), ('network', [network])):
            for name in names:
                result = subprocess.run(['docker', kind, 'inspect', name], capture_output=True, text=True)
                if result.returncode == 0 and json.loads(result.stdout)[0].get('Labels', {}).get('recserve.proof') == token:
                    subprocess.run(['docker', kind, 'rm', name], stdout=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    main()
