"""Test-only static mount. Production static files are served by Caddy."""
from pathlib import Path
from starlette.staticfiles import StaticFiles
from recserve_app.api import create_app


def build():
    app = create_app()
    app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[2] / 'web/dist', html=True))
    return app
