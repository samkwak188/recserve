"""Immutable identity and training-policy inputs from a verified bundle."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.bundle import member, sha256, verify


class Model:
    def __init__(self, manifest):
        manifest = Path(manifest).resolve()
        bundle = verify(manifest)
        self.fingerprint = sha256(manifest)
        self.version = bundle['model_version']
        self.root, self.bundle = manifest.parent, bundle

        def rows(role):
            with member(self.root, bundle[role]).open(newline='', encoding='utf-8') as source:
                return list(csv.DictReader(source))

        self.users = {r['user_id']: int(r['query_row']) for r in rows('user_map')}
        self.items = [r['item_id'] for r in rows('item_map')]
        self.item_rows = {item: row for row, item in enumerate(self.items)}
        meta_name = bundle.get('training_meta')
        if meta_name not in bundle['files']:
            raise ValueError('pilot requires verified training metadata and seen history')
        meta = json.loads(member(self.root, meta_name).read_text(encoding='utf-8'))
        train = meta['train_csv']
        if train not in bundle['files']:
            raise ValueError('unverified seen-history file')
        inverse_users = {row: user for user, row in self.users.items()}
        self.seen = defaultdict(set)
        popularity = Counter()
        with member(self.root, train).open(newline='', encoding='utf-8') as source:
            for row in csv.DictReader(source):
                user, item = int(row['query_row']), int(row['item'])
                if user not in inverse_users or not 0 <= item < len(self.items):
                    raise ValueError('training identity outside bundle')
                raw_user, raw_item = inverse_users[user], self.items[item]
                if raw_item not in self.seen[raw_user]:
                    popularity[raw_item] += 1
                    self.seen[raw_user].add(raw_item)
        # This is popularity among exported training users, not all MovieLens users.
        self.popular = sorted(self.items, key=lambda item: (-popularity[item], self.item_rows[item]))


def eligible(item, training_seen, blocked, unavailable):
    """Shared by the online policy and independent replay/quality callers."""
    return item not in training_seen and item not in blocked and item not in unavailable


def rank(candidates, seen, snapshot, item_rows):
    """Transparent pilot heuristic, not a learned ranker or a quality-lift claim."""
    output = []
    for item, score in candidates.items():
        if eligible(item, seen, snapshot['blocked'], snapshot['unavailable']):
            shown, saved = snapshot['features'].get(item, (0, 0))
            output.append((item, score + 0.05 * saved / (shown + 2)))
    return sorted(output, key=lambda row: (-row[1], item_rows[row[0]]))
