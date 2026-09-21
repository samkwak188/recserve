"""Test-only static mount. Production static files are served by Caddy."""
from pathlib import Path
from starlette.staticfiles import StaticFiles
from recserve_app.api import create_app
import os
from support import FileLedger


def build():
    app = create_app(ledger=FileLedger(os.environ['BROWSER_LEDGER_DIRECTORY']))
    app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[2] / 'web/dist', html=True))
    return app
