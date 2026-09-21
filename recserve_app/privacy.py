"""Encrypted, off-database deletion ledger. No plaintext account identifiers in logs."""
import json
import os
from urllib.parse import urlsplit
import uuid
import boto3
from botocore.config import Config
from cryptography.fernet import Fernet
from .db import now_ms


class S3DeletionLedger:
    def __init__(self, client, bucket, key):
        self.client, self.bucket, self.cipher = client, bucket, Fernet(key)

    @classmethod
    def from_env(cls):
        endpoint = os.environ.get('PRIVACY_S3_ENDPOINT')
        if not endpoint:
            return None
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError('Privacy ledger requires an HTTPS endpoint')
        client = boto3.client('s3', endpoint_url=endpoint, region_name=os.environ['PRIVACY_S3_REGION'],
            aws_access_key_id=os.environ['PRIVACY_S3_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['PRIVACY_S3_SECRET_ACCESS_KEY'],
            config=Config(connect_timeout=2, read_timeout=3, retries={'total_max_attempts': 2},
                          s3={'addressing_style': 'path'}))
        return cls(client, os.environ['PRIVACY_S3_BUCKET'], os.environ['PRIVACY_FERNET_KEY'].encode())

    def record(self, user_id, requested_ms=None):
        user_id = str(uuid.UUID(user_id))
        body = json.dumps(dict(schema=1, user_id=user_id, requested_ms=requested_ms or now_ms())).encode()
        self.client.put_object(Bucket=self.bucket, Key='deletions/' + user_id,
                               Body=self.cipher.encrypt(body), ContentType='application/octet-stream')

    def read_all(self):
        result = []
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix='deletions/'):
            for entry in page.get('Contents', []):
                if len(result) >= 10000 or entry.get('Size', 0) > 4096:
                    raise ValueError('Ledger bounds exceeded')
                response = self.client.get_object(Bucket=self.bucket, Key=entry['Key'])
                with response['Body'] as stream:
                    encrypted = stream.read(4097)
                if len(encrypted) > 4096:
                    raise ValueError('Oversized deletion record')
                body = json.loads(self.cipher.decrypt(encrypted))
                if body.get('schema') != 1 or str(uuid.UUID(body['user_id'])) != entry['Key'].removeprefix('deletions/'):
                    raise ValueError('Invalid deletion ledger record')
                result.append(body['user_id'])
        return result


def reconcile(db, ledger):
    """Must succeed before readiness after every restart/restore."""
    from sqlalchemy import select, update
    from .schema_v1 import outbox
    with db.engine.connect() as tx:
        pending = tx.execute(select(outbox).where(outbox.c.kind == 'delete', outbox.c.delivered_ms.is_(None))).mappings().all()
    for row in pending:
        ledger.record(row['user_id'], row['created_ms'])
        with db.engine.begin() as tx:
            tx.execute(update(outbox).where(outbox.c.id == row['id']).values(delivered_ms=now_ms()))
    db.replay_deletions(ledger.read_all())
    db.maintenance()
