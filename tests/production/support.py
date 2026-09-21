"""Explicit dependency injection for test-only independent privacy storage."""
import json
from pathlib import Path


class MemoryLedger:
    def __init__(self):
        self.records = set()

    def record(self, uid, requested_ms=None):
        self.records.add(uid)

    def read_all(self):
        return sorted(self.records)


class FileLedger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)

    def record(self, uid, requested_ms=None):
        (self.directory / uid).write_text(json.dumps({'user_id': uid}))

    def read_all(self):
        return [json.loads(path.read_text())['user_id'] for path in self.directory.iterdir()]
