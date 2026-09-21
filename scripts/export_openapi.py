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
schema = app.openapi()
if '--check' in sys.argv:
    if not destination.exists() or json.loads(destination.read_text()) != schema:
        raise SystemExit('OpenAPI contract is stale; regenerate it')
else:
    destination.write_text(json.dumps(schema, indent=2, sort_keys=True) + '\n')
app.state.db.engine.dispose()
print(destination)
