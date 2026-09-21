"""Operator-only synthetic seeding for isolated container proofs; never an HTTP route."""
import json
import os
import sys
sys.path.insert(0, '/app')
from sqlalchemy import insert, text
from recserve_app.db import Database
from recserve_app.entrypoint import secrets
from recserve_app.schema_v1 import invitations

secrets()
db = Database(os.environ['DATABASE_URL'])
with db.engine.begin() as tx:
    tx.execute(insert(invitations), [{'email': 'stack-one@example.invalid'}, {'email': 'stack-two@example.invalid'}])
    assert not tx.scalar(text("SELECT has_schema_privilege(current_user, 'public', 'CREATE')"))
    assert not tx.scalar(text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user"))
accounts = []
for name in ('one', 'two'):
    token, csrf = db.login('stack-' + name, 'stack-' + name + '@example.invalid')
    accounts.append(dict(token=token, csrf=csrf))
db.engine.dispose()
print(json.dumps(accounts))
