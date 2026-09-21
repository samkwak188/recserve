#!/usr/bin/env python3
"""Generate the frontend contract without connecting to a database or provider."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recserve_app.api import create_app
from recserve_app.config import Settings

app = create_app(Settings('postgresql+psycopg://schema:schema@127.0.0.1/schema',
                          'https://schema.invalid', 'schema-client', 'schema-secret'))
destination = ROOT / 'web/openapi.json'
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + '\n')
app.state.db.engine.dispose()
print(destination)
