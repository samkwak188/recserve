import os
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text, insert
from recserve_app.api import create_app, SESSION
from recserve_app.config import Settings
from recserve_app.schema_v1 import metadata, invitations
from recserve_app.model import Model
from recserve_app.policy import Policy
from recserve_app.retrieval import Retrieval
from scripts.model_v2 import write_bundle

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='session')
def live_model(tmp_path_factory):
    root = tmp_path_factory.mktemp('model-v2')
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(640, 8)).astype(np.float32)
    movies = [dict(id=i + 1, title=f'Test movie {i + 1}', year=2000, genres=['Drama'], available=True) for i in range(640)]
    manifest = write_bundle(root / 'bundle', raw, movies, list(range(1, 641)), 'als',
        {'dataset': 'synthetic-test-only'}, 'Synthetic fixture; no external data.', ROOT / 'build-production/recserve_fixture')
    model = Model(manifest)
    port, metrics = free_port(), free_port()
    while metrics == port:
        metrics = free_port()
    with (root / 'server.log').open('w') as output:
        process = subprocess.Popen([str(ROOT / 'build-production/recserve_server'), '--vectors-only',
            '--catalog', str(model.root / 'catalog.bin'), '--index', str(model.root / 'index.bin'),
            '--model-digest', model.digest, '--port', str(port), '--metrics-port', str(metrics)], stdout=output, stderr=output)
        upstream = Retrieval('127.0.0.1', port, model.digest, model.dimension)
        try:
            for _ in range(100):
                try:
                    upstream.ready()
                    break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError('Vector server exited')
                    time.sleep(.02)
            else:
                raise RuntimeError('Vector server readiness timeout')
            yield model, upstream
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture
def service(app, live_model):
    model, upstream = live_model
    service = Policy(app.state.db, model, upstream, app.state.settings.consent_version)
    app.state.policy = service
    return service


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
