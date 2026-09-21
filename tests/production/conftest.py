import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text, insert
from recserve_app.api import create_app, SESSION
from recserve_app.config import Settings
from recserve_app.schema_v1 import metadata, invitations


@pytest.fixture
def app():
    settings = Settings(os.environ['DATABASE_URL'], 'https://pilot.test', 'test-client', 'test-secret')
    app = create_app(settings)
    with app.state.db.engine.begin() as tx:
        for table in reversed(metadata.sorted_tables):
            tx.execute(table.delete())
        tx.execute(insert(invitations), [{'email': 'one@example.invalid'}, {'email': 'two@example.invalid'}])
    yield app
    app.state.db.engine.dispose()


@pytest.fixture
def client(app):
    with TestClient(app, base_url='https://pilot.test', follow_redirects=False) as client:
        yield client


def signed_in(app, client, number='one'):
    token, csrf = app.state.db.login('subject-' + number, number + '@example.invalid')
    client.cookies.set(SESSION, token, domain='pilot.test', path='/')
    client.headers.update({'origin': 'https://pilot.test', 'x-csrf-token': csrf})
    return app.state.db.authenticate(token), token, csrf
