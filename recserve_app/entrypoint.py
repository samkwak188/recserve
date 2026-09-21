"""Container entry points; read scoped secrets from mounted files, never argv."""
import os
from pathlib import Path
import sys


def secrets():
    for name in ('DATABASE_URL', 'MIGRATION_DATABASE_URL', 'GOOGLE_CLIENT_SECRET',
                 'PRIVACY_S3_ACCESS_KEY_ID', 'PRIVACY_S3_SECRET_ACCESS_KEY', 'PRIVACY_FERNET_KEY'):
        path = os.environ.get(name + '_FILE')
        if path:
            source = Path(path)
            if not source.is_file() or source.stat().st_size > 16384:
                raise RuntimeError('Invalid mounted secret file')
            os.environ[name] = source.read_text().strip()


def migrate():
    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from alembic import command
    from alembic.config import Config
    runtime = make_url(os.environ['DATABASE_URL'])
    owner = make_url(os.environ['MIGRATION_DATABASE_URL'])
    if runtime.username != 'recserve_app' or runtime.database != owner.database or not runtime.password:
        raise ValueError('Expected a separate recserve_app runtime role in the migration database')
    dsn = owner.set(drivername='postgresql').render_as_string(hide_password=False)
    with psycopg.connect(dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = 'recserve_app'")
            if cursor.fetchone() is None:
                cursor.execute(sql.SQL('CREATE ROLE recserve_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}').format(sql.Literal(runtime.password)))
    os.environ['DATABASE_URL'] = owner.render_as_string(hide_password=False)
    try:
        command.upgrade(Config('alembic.ini'), 'head')
    finally:
        os.environ['DATABASE_URL'] = runtime.render_as_string(hide_password=False)
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute('GRANT USAGE ON SCHEMA public TO recserve_app')
        connection.execute('GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO recserve_app')
        connection.execute('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO recserve_app')
        connection.execute('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO recserve_app')
    print('Migration and restricted runtime grants complete')


def main():
    secrets()
    action = sys.argv[1] if len(sys.argv) > 1 else 'api'
    if action == 'api':
        import uvicorn
        uvicorn.run('recserve_app.api:create_app', factory=True, host='0.0.0.0', port=8000,
            access_log=False, limit_concurrency=64, timeout_keep_alive=5, timeout_graceful_shutdown=10,
            proxy_headers=True, forwarded_allow_ips=os.environ.get('TRUSTED_PROXY_NETWORK', '127.0.0.1'))
    elif action == 'migrate':
        migrate()
    elif action == 'retrieval':
        from .model import Model
        model = Model(os.environ['MODEL_BUNDLE'])
        executable = '/usr/local/bin/recserve_server'
        os.execv(executable, [executable, '--vectors-only', '--catalog', str(model.root / 'catalog.bin'),
            '--index', str(model.root / 'index.bin'), '--model-digest', model.digest,
            '--bind', '0.0.0.0', '--workers', '4', '--queue', '32'])
    else:
        raise SystemExit('Unknown entrypoint action')


if __name__ == '__main__':
    main()
