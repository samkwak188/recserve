import time
from urllib.parse import parse_qs, urlsplit
import pytest
from sqlalchemy import select, update
from recserve_app.api import SESSION
from recserve_app.db import hashed, now_ms
from recserve_app.schema_v1 import sessions, users, identities, outbox
from conftest import signed_in


def test_account_consent_export_delete(app, client):
    assert client.get('/api/v2/me').status_code == 401
    actor, token, csrf = signed_in(app, client)
    me = client.get('/api/v2/me').json()
    assert me['id'] == actor.user_id and me['consent_version'] is None
    assert client.put('/api/v2/me/consent', json={'adult': True, 'research': True, 'user_id': 'forged'}).status_code == 422
    assert client.put('/api/v2/me/consent', json={'adult': True, 'research': True}).status_code == 200
    assert client.get('/api/v2/me/export').json()['identities'][0]['subject'] == 'subject-one'
    assert client.delete('/api/v2/me').status_code == 204
    client.cookies.set(SESSION, token, domain='pilot.test', path='/')
    assert client.get('/api/v2/me').status_code == 401
    with app.state.db.engine.connect() as tx:
        assert tx.execute(select(users)).first() is None
        assert tx.execute(select(identities)).first() is None
        assert tx.execute(select(outbox.c.user_id)).scalar_one() == actor.user_id


def test_csrf_origin_and_logout(app, client):
    _, token, _ = signed_in(app, client)
    client.headers['origin'] = 'https://attacker.invalid'
    assert client.post('/auth/logout').status_code == 403
    client.headers['origin'] = 'https://pilot.test'
    client.headers['x-csrf-token'] = 'wrong'
    assert client.post('/auth/logout').status_code == 403
    _, token, _ = signed_in(app, client)
    assert client.post('/auth/logout').status_code == 204
    client.cookies.set(SESSION, token, domain='pilot.test', path='/')
    assert client.get('/api/v2/me').status_code == 401


@pytest.mark.parametrize('column,age', [('seen_ms', 86400001), ('created_ms', 604800001)])
def test_expiry(app, client, column, age):
    _, token, _ = signed_in(app, client)
    with app.state.db.engine.begin() as tx:
        tx.execute(update(sessions).where(sessions.c.token_hash == hashed(token)).values({column: now_ms() - age}))
    assert client.get('/api/v2/me').status_code == 401


def test_ownership_and_session_hashing(app, client):
    first, token, _ = signed_in(app, client)
    second, _, _ = signed_in(app, client, 'two')
    assert first.user_id != second.user_id
    assert client.get('/api/v2/me/export').json()['identities'][0]['subject'] == 'subject-two'
    assert client.get('/api/v2/me/' + first.user_id).status_code == 404
    with app.state.db.engine.connect() as tx:
        stored = tx.execute(select(sessions.c.token_hash)).scalars().all()
        assert token not in stored and hashed(token) in stored


def test_login_state_replay_and_invitation(app, client):
    response = client.get('/auth/login')
    query = parse_qs(urlsplit(response.headers['location']).query)
    assert query['scope'] == ['openid email'] and query['code_challenge_method'] == ['S256']
    assert client.get('/auth/callback?state=wrong&code=x').status_code == 400
    async def identity(code, nonce, verifier):
        return 'subject-one', 'one@example.invalid'
    app.state.oidc.exchange = identity
    url = '/auth/callback?code=example&state=' + query['state'][0]
    response = client.get(url)
    assert response.status_code == 303
    assert 'HttpOnly' in response.headers.get_list('set-cookie')[1]
    assert client.get(url).status_code == 400
    response = client.get('/auth/login')
    state = parse_qs(urlsplit(response.headers['location']).query)['state'][0]
    async def uninvited(code, nonce, verifier):
        return 'outsider', 'outsider@example.invalid'
    app.state.oidc.exchange = uninvited
    assert client.get('/auth/callback', params={'code': 'x', 'state': state}).status_code == 403


def test_bounds_headers_and_rate_limit(app, client):
    signed_in(app, client)
    assert client.put('/api/v2/me/consent', content=b'x' * 32769).status_code == 413
    assert client.get('/healthz', headers={'host': 'attacker.invalid'}).status_code == 400
    assert client.get('/api/v2/me').headers['cache-control'] == 'no-store'
    assert client.get('/readyz').status_code == 503  # no model configured in account-only fixture
    for _ in range(10):
        assert client.get('/auth/login').status_code == 303
    assert client.get('/auth/login').status_code == 429
