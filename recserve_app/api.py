import asyncio
from contextlib import asynccontextmanager
import secrets
from urllib.parse import urlsplit

import anyio
from fastapi import FastAPI, Depends, HTTPException, Request, Response, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict
from typing import Literal
from sqlalchemy import select, update, delete, text
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings
from .db import Database, Principal, hashed, now_ms
from .oidc import GoogleOIDC
from .schema_v1 import users, sessions, preferences, item_states
from .privacy import S3DeletionLedger, reconcile
from .telemetry import Metrics, Telemetry

SESSION = '__Host-recserve'
CSRF = '__Host-recserve-csrf'
FLOW = '__Host-recserve-flow'


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Consent(StrictModel):
    adult: Literal[True]
    research: Literal[True]


class Bounds:
    """Bound admission and full-body reads before the framework parses JSON."""
    def __init__(self, app):
        self.app, self.active = app, 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if self.active >= 64:
            return await JSONResponse({'detail': 'Service busy'}, 503)(scope, receive, send)
        self.active += 1
        try:
            body = bytearray()
            deadline = asyncio.get_running_loop().time() + 5
            while True:
                try:
                    message = await asyncio.wait_for(receive(), max(0.001, deadline - asyncio.get_running_loop().time()))
                except asyncio.TimeoutError:
                    return await JSONResponse({'detail': 'Request timeout'}, 408)(scope, receive, send)
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > 32768:
                    return await JSONResponse({'detail': 'Body too large'}, 413)(scope, receive, send)
                if not message.get('more_body'):
                    break
            sent = False

            async def bounded_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
                return await receive()

            async def secure_send(message):
                if message['type'] == 'http.response.start':
                    csp = (b"default-src 'none'; frame-ancestors 'none'" if scope['path'].startswith(('/api/', '/auth/'))
                           else b"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
                    message.setdefault('headers', []).extend([
                        (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                        (b'referrer-policy', b'no-referrer'), (b'x-frame-options', b'DENY'),
                        (b'content-security-policy', csp),
                        (b'strict-transport-security', b'max-age=31536000')])
                await send(message)
            await self.app(scope, bounded_receive, secure_send)
        finally:
            self.active -= 1


def principal(request: Request) -> Principal:
    if 'user_id' in request.query_params:
        raise HTTPException(422, 'Identity is derived from the session')
    db = request.app.state.db
    result = db.authenticate(request.cookies.get(SESSION, ''))
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        if request.headers.get('origin') != request.app.state.settings.origin:
            raise HTTPException(403, 'Same-origin request required')
        if not secrets.compare_digest(hashed(request.headers.get('x-csrf-token', '')), result.csrf_hash):
            raise HTTPException(403, 'Invalid CSRF token')
        db.rate('mutation:' + result.user_id, 60)
    return result


def create_app(settings: Settings | None = None, ledger=None):
    settings = settings or Settings.from_env()
    db = Database(settings.database_url)
    ledger = ledger if ledger is not None else S3DeletionLedger.from_env()

    @asynccontextmanager
    async def lifespan(app):
        anyio.to_thread.current_default_thread_limiter().total_tokens = 16
        if ledger is None:
            raise RuntimeError('An independent deletion ledger is required before serving accounts')
        await anyio.to_thread.run_sync(reconcile, db, ledger)
        app.state.privacy_ready = True
        stop = asyncio.Event()

        async def maintenance():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=60)
                except asyncio.TimeoutError:
                    try:
                        await anyio.to_thread.run_sync(reconcile, db, ledger)
                        app.state.privacy_ready = True
                    except Exception:
                        app.state.privacy_ready = False

        task = asyncio.create_task(maintenance())
        try:
            yield
        finally:
            stop.set()
            await task
            db.engine.dispose()

    app = FastAPI(title='RecServe movie pilot', version='2.0.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.db, app.state.oidc = settings, db, GoogleOIDC(settings)
    app.state.ledger, app.state.privacy_ready = ledger, False
    app.state.metrics = Metrics()
    app.state.policy = None
    if settings.bundle:
        from .model import Model
        from .retrieval import Retrieval
        from .policy import Policy
        model = Model(settings.bundle)
        app.state.policy = Policy(db, model, Retrieval(settings.upstream_host, settings.upstream_port,
                                 model.digest, model.dimension), settings.consent_version)
    app.add_middleware(Bounds)
    app.add_middleware(Telemetry, metrics=app.state.metrics)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlsplit(settings.origin).hostname])
    from .contracts import (Preferences, Recommendation, Event, Me, MoviePage, PreferencePage,
        PreferenceResult, RecommendationResult, Watchlist, EventResult)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        # Do not expose SQL, credentials, or user payloads through exception text.
        return JSONResponse({'detail': 'Database unavailable'}, 503)

    @app.get('/healthz')
    def health():
        return {'status': 'alive'}

    @app.get('/internal/metrics', include_in_schema=False)
    def metrics():
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(generate_latest(app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get('/readyz')
    def ready():
        if not app.state.privacy_ready:
            raise HTTPException(503, 'Privacy ledger replay unavailable')
        with db.engine.connect() as connection:
            revision = connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one()
        if revision != '0001':
            raise HTTPException(503, 'Unsupported database revision')
        if app.state.policy is None:
            raise HTTPException(503, 'Model not configured')
        try:
            app.state.policy.retrieval.ready()
        except (OSError, ValueError, EOFError):
            raise HTTPException(503, 'Retrieval not ready') from None
        return {'status': 'ready', 'schema': revision}

    @app.get('/auth/login')
    async def login(request: Request):
        address = request.client.host if request.client else 'unknown'
        await anyio.to_thread.run_sync(db.rate, 'login:' + hashed(address), 10)
        state, browser, nonce, verifier = await anyio.to_thread.run_sync(db.start_flow)
        response = RedirectResponse(await app.state.oidc.authorize_url(state, nonce, verifier), 303)
        response.set_cookie(FLOW, browser, max_age=600, secure=True, httponly=True, samesite='lax')
        return response

    @app.get('/auth/callback')
    async def callback(request: Request, state: str = '', code: str = ''):
        if not state or not code or len(state) > 128 or len(code) > 4096:
            raise HTTPException(400, 'Invalid login response')
        flow = await anyio.to_thread.run_sync(db.consume_flow, state, request.cookies.get(FLOW, ''))
        try:
            subject, email = await app.state.oidc.exchange(code, flow['nonce'], flow['verifier'])
        except Exception:
            raise HTTPException(400, 'Identity verification failed') from None
        token, csrf = await anyio.to_thread.run_sync(db.login, subject, email)
        response = RedirectResponse('/', 303)
        response.delete_cookie(FLOW, secure=True, httponly=True, samesite='lax')
        response.set_cookie(SESSION, token, max_age=604800, secure=True, httponly=True, samesite='lax')
        response.set_cookie(CSRF, csrf, max_age=604800, secure=True, httponly=False, samesite='lax')
        return response

    @app.post('/auth/logout', status_code=204)
    def logout(response: Response, actor: Principal = Depends(principal)):
        with db.engine.begin() as tx:
            tx.execute(delete(sessions).where(sessions.c.token_hash == actor.token_hash))
        response.delete_cookie(SESSION, secure=True, httponly=True, samesite='lax')
        response.delete_cookie(CSRF, secure=True, samesite='lax')

    @app.get('/api/v2/me', response_model=Me)
    def me(actor: Principal = Depends(principal)):
        with db.engine.begin() as tx:
            row = db.lock_user(tx, actor.user_id)
            return {'id': actor.user_id, 'consent_version': row['consent_version'],
                    'required_consent': settings.consent_version, 'preference_revision': row['revision']}

    @app.put('/api/v2/me/consent')
    def consent(payload: Consent, actor: Principal = Depends(principal)):
        with db.engine.begin() as tx:
            db.lock_user(tx, actor.user_id)
            tx.execute(update(users).where(users.c.id == actor.user_id).values(
                consent_version=settings.consent_version, consent_ms=now_ms()))
        return {'consent_version': settings.consent_version}

    @app.get('/api/v2/me/export')
    def export(actor: Principal = Depends(principal)):
        return db.export(actor.user_id)

    @app.delete('/api/v2/me', status_code=204)
    def remove(response: Response, actor: Principal = Depends(principal)):
        if app.state.ledger is None:
            raise HTTPException(503, 'Deletion ledger unavailable')
        try:
            db.delete_account(actor.user_id, app.state.ledger.record)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, 'Deletion could not be confirmed; retry or contact the operator') from None
        response.delete_cookie(SESSION, secure=True, httponly=True, samesite='lax')
        response.delete_cookie(CSRF, secure=True, samesite='lax')

    def policy():
        if app.state.policy is None:
            raise HTTPException(503, 'Model not configured')
        return app.state.policy

    @app.get('/api/v2/movies', response_model=MoviePage)
    def movies(query: str = Query('', max_length=100), cursor: int = Query(0, ge=0, le=100000),
               actor: Principal = Depends(principal)):
        service = policy()
        with db.engine.begin() as tx:
            service.account(tx, actor.user_id)
        found = [m for m in service.model.movies if m['available'] and query.casefold() in m['title'].casefold()]
        return {'items': found[cursor:cursor + 20], 'next_cursor': cursor + 20 if cursor + 20 < len(found) else None}

    @app.get('/api/v2/preferences', response_model=PreferencePage)
    def get_preferences(actor: Principal = Depends(principal)):
        service = policy()
        with db.engine.begin() as tx:
            user = service.account(tx, actor.user_id)
            items = [dict(row) for row in tx.execute(select(preferences.c.item_id, preferences.c.value).where(
                preferences.c.user_id == actor.user_id)).mappings()]
        return {'items': items, 'preference_revision': user['revision']}

    @app.put('/api/v2/preferences', response_model=PreferenceResult)
    def set_preferences(payload: Preferences, actor: Principal = Depends(principal)):
        return policy().change_preferences(actor.user_id, payload)

    @app.post('/api/v2/recommendations', response_model=RecommendationResult)
    def recommendations(payload: Recommendation, actor: Principal = Depends(principal)):
        result = policy().recommend(actor.user_id, payload)
        app.state.metrics.recommendations.labels(result['candidate_source'], str(result['degraded']).lower()).inc()
        return result

    @app.post('/api/v2/events', response_model=EventResult)
    def feedback(payload: Event, actor: Principal = Depends(principal)):
        return policy().event(actor.user_id, payload)

    @app.get('/api/v2/watchlist', response_model=Watchlist)
    def watchlist(actor: Principal = Depends(principal)):
        service = policy()
        with db.engine.begin() as tx:
            service.account(tx, actor.user_id)
            states = tx.execute(select(item_states).where(item_states.c.user_id == actor.user_id)).mappings().all()
        return {'items': [dict(service.model.movies[service.model.rows[row['item_id']]],
            saved=row['saved'], watched=row['watched'], dismissed=row['dismissed'])
            for row in states if row['item_id'] in service.model.rows]}

    return app
