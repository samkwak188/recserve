#!/usr/bin/env python3
"""Encrypted pgBackRest crash/restore proof on isolated local Docker volumes."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.production_runner import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    image = json.loads(args.images.read_text())['images']['database']['id']
    token = uuid.uuid4().hex[:12]
    prefix = 'recserve-recovery-' + token
    primary, restored = prefix + '-primary', prefix + '-restored'
    volumes = [prefix + '-' + role for role in ('data', 'repository', 'restored')]
    containers = []
    directory = ROOT / '.cache/production/recovery' / str(time.time_ns())
    directory.mkdir(parents=True)
    env = dict(os.environ, POSTGRES_PASSWORD=secrets.token_hex(24))
    started = time.monotonic()

    def run(command, capture=False, **kwargs):
        return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE if capture else None,
                              **kwargs).stdout

    def sql(container, query):
        return run(['docker', 'exec', '-i', container, 'psql', '-U', 'recserve_owner', '-d', 'recserve',
                    '-v', 'ON_ERROR_STOP=1', '-At'], capture=True, input=query).strip()

    def wait(container):
        for _ in range(120):
            result = subprocess.run(['docker', 'exec', container, 'pg_isready', '-U', 'recserve_owner', '-d', 'recserve'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                return
            time.sleep(.5)
        raise RuntimeError('PostgreSQL readiness timeout')

    with tempfile.TemporaryDirectory(prefix='secrets-', dir=directory) as temp:
        config = Path(temp) / 'pgbackrest.conf'
        config.write_text('[global]\nrepo1-type=posix\nrepo1-path=/var/lib/pgbackrest\n'
            'repo1-cipher-type=aes-256-cbc\nrepo1-cipher-pass=' + secrets.token_urlsafe(32) + '\n'
            'repo1-retention-full=2\nstart-fast=y\narchive-timeout=30\nlog-level-console=warn\n'
            'log-level-file=off\nlock-path=/tmp/pgbackrest\n[recserve]\npg1-path=/var/lib/postgresql/data\n'
            'pg1-user=recserve_owner\npg1-database=recserve\n')
        config.chmod(0o644)  # ephemeral secret readable by the isolated container's UID 999
        common = ['--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
            '--tmpfs', '/tmp:size=32m', '--tmpfs', '/var/run/postgresql:uid=999,gid=999,mode=3775,size=8m',
            '--label', 'recserve.proof=' + token,
            '--mount', f'type=bind,source={config},target=/etc/pgbackrest/pgbackrest.conf,readonly']
        try:
            for volume in volumes:
                run(['docker', 'volume', 'create', '--label', 'recserve.proof=' + token, volume])
            containers.append(primary)
            run(['docker', 'run', '-d', '--name', primary, *common, '-e', 'POSTGRES_PASSWORD',
                '-e', 'POSTGRES_USER=recserve_owner', '-e', 'POSTGRES_DB=recserve',
                '--mount', f'type=volume,source={volumes[0]},target=/var/lib/postgresql/data',
                '--mount', f'type=volume,source={volumes[1]},target=/var/lib/pgbackrest', image,
                'postgres', '-c', 'archive_mode=on', '-c', 'archive_timeout=60', '-c',
                'archive_command=pgbackrest --stanza=recserve archive-push %p'], env=env)
            wait(primary)
            sql(primary, 'CREATE TABLE proof_events (id integer PRIMARY KEY, created_at timestamptz DEFAULT clock_timestamp()); INSERT INTO proof_events(id) VALUES (1);')
            run(['docker', 'exec', primary, 'pgbackrest', '--stanza=recserve', 'stanza-create'])
            run(['docker', 'exec', primary, 'pgbackrest', '--stanza=recserve', '--type=full', 'backup'])
            sql(primary, 'INSERT INTO proof_events(id) VALUES (2);')
            committed_at = time.monotonic()
            run(['docker', 'exec', primary, 'pgbackrest', '--stanza=recserve', 'check'])
            archived_after_s = time.monotonic() - committed_at
            wrong_key = subprocess.run(['docker', 'exec', primary, 'pgbackrest', '--stanza=recserve',
                '--repo1-cipher-pass=deliberately-wrong-test-key', 'info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if wrong_key.returncode == 0:
                raise RuntimeError('Backup unexpectedly readable with an incorrect encryption key')
            run(['docker', 'kill', primary])
            recovery_start = time.monotonic()
            run(['docker', 'run', '--rm', *common,
                '--mount', f'type=volume,source={volumes[2]},target=/var/lib/postgresql/data',
                '--mount', f'type=volume,source={volumes[1]},target=/var/lib/pgbackrest,readonly',
                '--entrypoint', 'pgbackrest', image, '--stanza=recserve', 'restore'])
            containers.append(restored)
            run(['docker', 'run', '-d', '--name', restored, *common,
                '--mount', f'type=volume,source={volumes[2]},target=/var/lib/postgresql/data',
                '--mount', f'type=volume,source={volumes[1]},target=/var/lib/pgbackrest,readonly',
                image, 'postgres', '-c', 'archive_mode=off'])
            wait(restored)
            if sql(restored, 'SELECT string_agg(id::text, \',\' ORDER BY id) FROM proof_events;') != '1,2':
                raise RuntimeError('Committed pre/post-backup records were not recovered')
            if sql(restored, 'SELECT pg_is_in_recovery();') != 'f':
                raise RuntimeError('Restored server is not writable')
            restored_s = time.monotonic() - recovery_start
            report = dict(source=source_identity(ROOT), image=image, scope='local encrypted repository; not off-host/cloud evidence',
                records_recovered=2, wrong_encryption_key_rejected=True, archive_confirmation_s=archived_after_s,
                restore_s=restored_s, elapsed_s=time.monotonic() - started, off_host=False,
                postgres_version=sql(restored, 'SHOW server_version;'))
            (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report))
        finally:
            for name in containers:
                inspection = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
                if inspection.returncode == 0 and json.loads(inspection.stdout)[0]['Config'].get('Labels', {}).get('recserve.proof') == token:
                    with (directory / (name + '.log')).open('w') as output:
                        subprocess.run(['docker', 'logs', name], stdout=output, stderr=subprocess.STDOUT, check=False)
                    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, check=False)
            for volume in volumes:
                inspection = subprocess.run(['docker', 'volume', 'inspect', volume], capture_output=True, text=True)
                if inspection.returncode == 0 and json.loads(inspection.stdout)[0].get('Labels', {}).get('recserve.proof') == token:
                    subprocess.run(['docker', 'volume', 'rm', volume], stdout=subprocess.DEVNULL, check=False)


if __name__ == '__main__':
    main()
