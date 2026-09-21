from concurrent.futures import ThreadPoolExecutor
import json
import socket
import struct
import uuid
import numpy as np
import pytest
from sqlalchemy import select
from recserve_app.contracts import Preferences, Recommendation
from recserve_app.db import now_ms
from recserve_app.model import Model
from recserve_app.retrieval import Retrieval, MAGIC
from recserve_app.schema_v1 import users, preferences
from conftest import signed_in


def uid():
    return str(uuid.uuid4())


def consent(app, client, number='one'):
    actor, _, _ = signed_in(app, client, number)
    assert client.put('/api/v2/me/consent', json={'adult': True, 'research': True}).status_code == 200
    return actor


def test_fold_in_matches_full_confidence_reference(live_model):
    model, upstream = live_model
    prefs = {1: 1, 8: 1, 11: -1}
    y = model.raw.astype(np.float64)
    c, p = np.ones(len(y)), np.zeros(len(y))
    for item, value in prefs.items():
        c[model.rows[item]], p[model.rows[item]] = (41, 1) if value == 1 else (11, 0)
    expected = np.linalg.solve(y.T @ (c[:, None] * y) + model.regularization * np.eye(model.dimension), y.T @ (c * p))
    expected /= np.linalg.norm(expected)
    actual = model.fold_in(prefs, 'als')
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert model.fold_in({1: -1}, 'als') is None
    assert model.fold_in(prefs, 'popularity') is None
    query = model.fold_in(prefs, 'centroid')
    exact = np.argsort(-(model.vectors @ query))[:128]
    got = [row for row, _ in upstream.query(query, 128)]
    assert len(set(exact) & set(got)) / 128 >= .98


def test_protocol_rejects_bad_models_vectors_and_frames(live_model):
    model, upstream = live_model
    with pytest.raises(ValueError):
        Retrieval(upstream.host, upstream.port, '0' * 64, model.dimension).ready()
    with pytest.raises(ValueError):
        Retrieval(upstream.host, upstream.port, model.digest, model.dimension - 1).ready()
    with pytest.raises(EOFError):
        upstream.query([float('nan')] * model.dimension, 5)
    with socket.create_connection((upstream.host, upstream.port), timeout=2) as connection:
        connection.sendall(struct.pack('<II', MAGIC, 0xffffffff))
        assert connection.recv(1) == b''
    with socket.create_connection((upstream.host, upstream.port), timeout=2) as connection:
        connection.sendall(b'R')
        assert connection.recv(1) == b''
    upstream.ready()


def test_personalization_feedback_and_retry(service, app, client):
    actor = consent(app, client)
    change = dict(request_id=uid(), changes=[dict(item_id=1, value=1), dict(item_id=3, value=-1)])
    changed = client.put('/api/v2/preferences', json=change)
    assert changed.status_code == 200
    assert client.put('/api/v2/preferences', json=change).json() == changed.json()
    assert client.put('/api/v2/preferences', json={**change, 'changes': [dict(item_id=2, value=1)]}).status_code == 409
    request = dict(request_id=uid(), k=10)
    response = client.post('/api/v2/recommendations', json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['candidate_source'] == 'als' and result['preference_revision'] == 1
    assert {1, 3}.isdisjoint(item['id'] for item in result['items'])
    assert client.post('/api/v2/recommendations', json=request).json() == result
    assert client.post('/api/v2/recommendations', json={**request, 'k': 9}).status_code == 409
    event = dict(event_id=uid(), request_id=request['request_id'], item_id=result['items'][0]['id'], kind='save', event_time_ms=now_ms())
    assert client.post('/api/v2/events', json=event).status_code == 409
    shown = {**event, 'event_id': uid(), 'kind': 'shown'}
    assert client.post('/api/v2/events', json=shown).status_code == 200
    assert client.post('/api/v2/events', json=event).status_code == 200
    assert client.post('/api/v2/events', json=event).json()['duplicate'] is True
    assert client.post('/api/v2/events', json={**event, 'event_id': uid()}).status_code == 409
    fresh = client.post('/api/v2/recommendations', json=dict(request_id=uid(), k=10)).json()
    assert fresh['preference_revision'] == 2
    assert event['item_id'] not in [item['id'] for item in fresh['items']]
    assert client.get('/api/v2/watchlist').json()['items'][0]['saved'] is True
    assert client.get('/readyz').status_code == 200


def test_cross_account_feedback_and_cold_start(service, app, client):
    consent(app, client)
    request = dict(request_id=uid(), k=3)
    first = client.post('/api/v2/recommendations', json=request).json()
    assert first['candidate_source'] == 'popularity'
    consent(app, client, 'two')
    assert client.post('/api/v2/events', json=dict(event_id=uid(), request_id=request['request_id'],
        item_id=first['items'][0]['id'], kind='shown', event_time_ms=now_ms())).status_code == 422
    assert client.get('/api/v2/watchlist').json()['items'] == []
    assert client.post('/api/v2/recommendations', json={**request, 'user_id': 'forged'}).status_code == 422


def test_concurrent_writes_and_snapshot_fence(service, app, client, monkeypatch):
    actor = consent(app, client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: service.change_preferences(actor.user_id,
            Preferences(request_id=uid(), changes=[{'item_id': i, 'value': 1}])), range(1, 9)))
    assert {r['preference_revision'] for r in results} == set(range(1, 9))
    original = service.retrieval.query
    raced = False
    def query(*args):
        nonlocal raced
        if not raced:
            raced = True
            service.change_preferences(actor.user_id, Preferences(request_id=uid(), changes=[{'item_id': 9, 'value': -1}]))
        return original(*args)
    monkeypatch.setattr(service.retrieval, 'query', query)
    response = service.recommend(actor.user_id, Recommendation(request_id=uid(), k=20))
    assert response['preference_revision'] == 9
    assert set(range(1, 10)).isdisjoint(item['id'] for item in response['items'])


def test_retrieval_failure_filters_and_database_failure_closes(service, app, client, monkeypatch):
    consent(app, client)
    assert client.put('/api/v2/preferences', json=dict(request_id=uid(), changes=[{'item_id': 1, 'value': 1}])).status_code == 200
    def fail(*args):
        raise TimeoutError('fixture')
    monkeypatch.setattr(service.retrieval, 'query', fail)
    result = client.post('/api/v2/recommendations', json=dict(request_id=uid(), k=3)).json()
    assert result['degraded'] is True and 1 not in [item['id'] for item in result['items']]
    from sqlalchemy.exc import OperationalError
    def dbfail(*args):
        raise OperationalError('fixture', {}, Exception('fixture'))
    monkeypatch.setattr(app.state.db, 'authenticate', dbfail)
    assert client.post('/api/v2/recommendations', json=dict(request_id=uid(), k=3)).status_code == 503


def test_transaction_rolls_back_unknown_item(service, app, client):
    actor = consent(app, client)
    response = client.put('/api/v2/preferences', json=dict(request_id=uid(), changes=[
        {'item_id': 1, 'value': 1}, {'item_id': 999999, 'value': 1}]))
    assert response.status_code == 422
    assert client.get('/api/v2/preferences').json() == {'items': [], 'preference_revision': 0}
