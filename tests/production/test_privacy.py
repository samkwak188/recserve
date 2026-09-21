import io
import uuid
import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber, ANY
from cryptography.fernet import Fernet
from sqlalchemy import select
from recserve_app.privacy import S3DeletionLedger, reconcile
from recserve_app.schema_v1 import users, identities
from conftest import signed_in


def test_acknowledged_deletion_survives_restore_replay(app, client):
    actor, token, _ = signed_in(app, client)
    with app.state.db.engine.connect() as tx:
        user = dict(tx.execute(select(users).where(users.c.id == actor.user_id)).mappings().one())
        identity = dict(tx.execute(select(identities).where(identities.c.user_id == actor.user_id)).mappings().one())
    assert client.delete('/api/v2/me').status_code == 204
    assert actor.user_id in app.state.ledger.read_all()
    # Simulate the older DB state restored independently of the external ledger.
    with app.state.db.engine.begin() as tx:
        tx.execute(users.insert().values(**user))
        tx.execute(identities.insert().values(**identity))
    reconcile(app.state.db, app.state.ledger)
    with app.state.db.engine.connect() as tx:
        assert tx.execute(select(users)).first() is None


def test_failed_offhost_write_does_not_acknowledge_deletion(app, client, monkeypatch):
    actor, _, _ = signed_in(app, client)
    def unavailable(*args):
        raise OSError('ledger unavailable')
    monkeypatch.setattr(app.state.ledger, 'record', unavailable)
    assert client.delete('/api/v2/me').status_code == 503
    assert client.get('/api/v2/me').status_code == 200


def test_encrypted_s3_ledger_and_tampering():
    client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='fixture', aws_secret_access_key='fixture')
    cipher_key = Fernet.generate_key()
    ledger = S3DeletionLedger(client, 'fixture-bucket', cipher_key)
    user_id = str(uuid.uuid4())
    import json
    body = Fernet(cipher_key).encrypt(json.dumps(dict(schema=1, user_id=user_id, requested_ms=1)).encode())
    key = 'deletions/' + user_id
    with Stubber(client) as stub:
        stub.add_response('put_object', {}, {'Bucket': 'fixture-bucket', 'Key': key, 'Body': ANY, 'ContentType': 'application/octet-stream'})
        ledger.record(user_id)
        stub.add_response('list_objects_v2', {'Contents': [{'Key': key, 'Size': len(body)}]}, {'Bucket': 'fixture-bucket', 'Prefix': 'deletions/'})
        stub.add_response('get_object', {'Body': StreamingBody(io.BytesIO(body), len(body))}, {'Bucket': 'fixture-bucket', 'Key': key})
        assert ledger.read_all() == [user_id]
        bad = body[:-4] + b'xxxx'
        stub.add_response('list_objects_v2', {'Contents': [{'Key': key, 'Size': len(bad)}]}, {'Bucket': 'fixture-bucket', 'Prefix': 'deletions/'})
        stub.add_response('get_object', {'Body': StreamingBody(io.BytesIO(bad), len(bad))}, {'Bucket': 'fixture-bucket', 'Key': key})
        with pytest.raises(Exception):
            ledger.read_all()


def test_metrics_have_no_user_identity_labels(app, client):
    actor, token, _ = signed_in(app, client)
    client.get('/api/v2/me')
    response = client.get('/internal/metrics')
    assert response.status_code == 200
    assert 'recserve_http_requests_total' in response.text
    assert actor.user_id not in response.text and token not in response.text
