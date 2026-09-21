#!/usr/bin/env python3
"""Isolated PostgreSQL integration suite; remove only this run's container."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    name = 'recserve-test-' + uuid.uuid4().hex[:12]
    password = secrets.token_hex(24)
    env = dict(os.environ, POSTGRES_PASSWORD=password)
    image = 'postgres:17.6-bookworm'
    subprocess.run(['docker', 'pull', image], check=True)
    try:
        subprocess.run(['docker', 'run', '-d', '--name', name, '--label', 'recserve.test=true',
            '-e', 'POSTGRES_PASSWORD', '-e', 'POSTGRES_DB=recserve_test',
            '-p', '127.0.0.1::5432', '--tmpfs', '/var/lib/postgresql/data', image], env=env, check=True)
        for _ in range(60):
            if subprocess.run(['docker', 'exec', name, 'pg_isready', '-U', 'postgres'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('PostgreSQL startup timed out')
        mapping = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        port = mapping['NetworkSettings']['Ports']['5432/tcp'][0]['HostPort']
        env['DATABASE_URL'] = f'postgresql+psycopg://postgres:{password}@127.0.0.1:{port}/recserve_test'
        subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', 'head'], cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/production', '-q', '--tb=short'],
                       cwd=ROOT, env=env, check=True)
        print('PostgreSQL migrations and production integration checks passed')
    finally:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    main()
