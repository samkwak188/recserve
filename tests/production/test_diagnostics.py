"""Real contention and wire faults must be diagnosable and recoverable."""
from contextlib import ExitStack
import socket
import struct
import threading
import time
import uuid

import pytest
from sqlalchemy import select

from conftest import signed_in
from recserve_app.retrieval import Retrieval, MAGIC
from recserve_app.schema_v1 import users
from recserve_app.telemetry import Metrics


def sample(app, name, labels=None):
    return app.state.metrics.registry.get_sample_value(name, labels or {})


def test_pool_exhaustion_is_bounded_and_recovers(app, client):
    signed_in(app, client)
    with ExitStack() as stack:
        for _ in range(8):
            stack.enter_context(app.state.db.engine.connect())
        metrics = client.get('/internal/metrics')
        assert metrics.status_code == 200
        assert 'recserve_database_pool_checked_out 8.0' in metrics.text
        assert 'recserve_database_pool_capacity 8.0' in metrics.text
        started = time.monotonic()
        response = client.get('/api/v2/me')
        assert response.status_code == 503
        assert response.json() == {'detail': 'Database unavailable'}
        assert time.monotonic() - started < 4
        assert sample(app, 'recserve_database_errors_total', {'reason': 'pool_timeout'}) == 1
    assert client.get('/api/v2/me').status_code == 200
    assert sample(app, 'recserve_database_pool_checked_out') == 0


def test_lock_timeout_rolls_back_then_retry_commits_once(service, app, client):
    actor, _, _ = signed_in(app, client)
    assert client.put('/api/v2/me/consent', json={'adult': True, 'research': True}).status_code == 200
    payload = {'request_id': str(uuid.uuid4()), 'changes': [{'item_id': 1, 'value': 1}]}
    with app.state.db.engine.begin() as blocker:
        blocker.execute(select(users).where(users.c.id == actor.user_id).with_for_update())
        started = time.monotonic()
        response = client.put('/api/v2/preferences', json=payload)
        assert response.status_code == 503
        assert time.monotonic() - started < 4
        assert sample(app, 'recserve_database_errors_total', {'reason': 'lock_timeout'}) == 1
    assert client.get('/api/v2/preferences').json() == {'items': [], 'preference_revision': 0}
    response = client.put('/api/v2/preferences', json=payload)
    assert response.status_code == 200
    assert response.json() == {'preference_revision': 1, 'count': 1}
    assert client.put('/api/v2/preferences', json=payload).json() == response.json()
    assert client.get('/api/v2/preferences').json()['preference_revision'] == 1
    assert sample(app, 'recserve_database_pool_checked_out') == 0


@pytest.mark.parametrize('fault,exception,outcome', [
    ('close', EOFError, 'connection'),
    ('frame', ValueError, 'protocol'),
    ('stall', TimeoutError, 'timeout'),
    ('trickle', TimeoutError, 'timeout'),
])
def test_wire_fault_diagnostics_and_absolute_deadline(fault, exception, outcome):
    metrics, stop, accepted = Metrics(), threading.Event(), threading.Event()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(2)
        port = listener.getsockname()[1]

        def serve():
            with listener.accept()[0] as peer:
                peer.settimeout(2)
                peer.recv(4096)
                accepted.set()
                if fault == 'frame':
                    peer.sendall(struct.pack('<II', MAGIC, 0xffffffff))
                elif fault == 'stall':
                    stop.wait(2)
                elif fault == 'trickle':
                    for byte in struct.pack('<II', MAGIC, 48):
                        if stop.wait(.02):
                            break
                        try:
                            peer.sendall(bytes([byte]))
                        except OSError:
                            break

        worker = threading.Thread(target=serve)
        worker.start()
        try:
            retrieval = Retrieval('127.0.0.1', port, '0' * 64, 1, metrics=metrics)
            started = time.monotonic()
            with pytest.raises(exception):
                retrieval.query([1.], 1, deadline=started + .1)
            assert accepted.wait(1)
            assert time.monotonic() - started < 1
            assert metrics.registry.get_sample_value('recserve_retrieval_duration_seconds_count', {'outcome': outcome}) == 1
        finally:
            stop.set()
            worker.join(3)
            assert not worker.is_alive()


def test_retrieval_success_and_expired_zero_deadline(live_model):
    model, upstream = live_model
    metrics = Metrics()
    retrieval = Retrieval(upstream.host, upstream.port, model.digest, model.dimension, metrics=metrics)
    retrieval.ready()
    with pytest.raises(TimeoutError):
        retrieval.query([0.] * model.dimension, 1, deadline=0)
    for outcome in ('success', 'timeout'):
        assert metrics.registry.get_sample_value('recserve_retrieval_duration_seconds_count', {'outcome': outcome}) == 1
