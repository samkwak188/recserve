"""Test-only crash injection; no environment-controlled crash hook in the API."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recserve_pilot.store import Store

if __name__ == '__main__':
    database, event_path, boundary = sys.argv[1:]
    def crash(point):
        if point == boundary:
            os._exit(77)
    Store(database, 'fixture-sha').apply(json.loads(Path(event_path).read_text()), crash=crash)
    raise RuntimeError('crash boundary not reached')
