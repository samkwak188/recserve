"""Verified item-only models and preference fold-in; no fixture-user identities."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct
import threading
import numpy as np


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Model:
    def __init__(self, path):
        path = Path(path).resolve()
        if path.stat().st_size > 65536:
            raise ValueError('Oversized manifest')
        self.digest = file_hash(path)
        self.manifest = info = json.loads(path.read_text())
        if info.get('schema_version') != 2 or info.get('policy') not in ('popularity', 'centroid', 'als'):
            raise ValueError('Unsupported bundle schema or policy')
        self.root = path.parent
        required = {'catalog.bin', 'factors.npy', 'gram.npy', 'index.bin', 'movies.json', 'popularity.json', 'LICENSE.txt'}
        if set(info['files']) != required:
            raise ValueError('Bundle roles do not match schema v2')
        for name, digest in info['files'].items():
            member = (self.root / name).resolve()
            if member.parent != self.root or member.stat().st_size > 512 * 1024 * 1024 or file_hash(member) != digest:
                raise ValueError('Unverified bundle member')
        with (self.root / 'catalog.bin').open('rb') as stream:
            magic, count, dim = struct.unpack('<Iii', stream.read(12))
        if magic != 0x43415431 or not 1 <= count <= 100000 or not 1 <= dim <= 256:
            raise ValueError('Invalid catalog dimensions')
        if (self.root / 'catalog.bin').stat().st_size != 12 + 4 * count * dim:
            raise ValueError('Invalid catalog length')
        self.vectors = np.memmap(self.root / 'catalog.bin', dtype='<f4', mode='r', offset=12, shape=(count, dim))
        self.raw = np.load(self.root / 'factors.npy', allow_pickle=False, mmap_mode='r')
        self.gram = np.load(self.root / 'gram.npy', allow_pickle=False)
        if self.raw.shape != (count, dim) or self.gram.shape != (dim, dim):
            raise ValueError('Incompatible factor basis')
        if not all(np.isfinite(array).all() for array in (self.vectors, self.raw, self.gram)):
            raise ValueError('Nonfinite model')
        raw64 = np.asarray(self.raw, dtype=np.float64)
        normalized = raw64 / np.maximum(np.linalg.norm(raw64, axis=1, keepdims=True), 1e-12)
        if not np.allclose(self.vectors, normalized, atol=1e-5) or not np.allclose(self.gram, raw64.T @ raw64, rtol=1e-5, atol=1e-5):
            raise ValueError('Factor/catalog/Gram mismatch')
        with (self.root / 'index.bin').open('rb') as stream:
            if struct.unpack('<Iii', stream.read(12)) != (0x48535732, count, dim):
                raise ValueError('Index dimensions differ')
        self.movies = json.loads((self.root / 'movies.json').read_text())
        if len(self.movies) != count or any(type(m.get('id')) is not int or m['id'] < 1 or
                not isinstance(m.get('title'), str) or len(m['title']) > 500 or
                type(m.get('available')) is not bool for m in self.movies):
            raise ValueError('Invalid movie map')
        self.rows = {movie['id']: row for row, movie in enumerate(self.movies)}
        if len(self.rows) != count:
            raise ValueError('Duplicate movie identities')
        self.popularity = json.loads((self.root / 'popularity.json').read_text())
        if len(self.popularity) != count or set(self.popularity) != set(self.rows):
            raise ValueError('Invalid popularity permutation')
        self.regularization = float(info['regularization'])
        if not 0 < self.regularization <= 100 or info.get('confidence') != {'like': 41, 'dislike': 11}:
            raise ValueError('Invalid fold-in contract')
        self.policy = info['policy']
        self.dimension = dim
        self._cache, self._lock, self._compute = OrderedDict(), threading.Lock(), threading.BoundedSemaphore(2)

    def vector(self, uid, revision, preferences):
        key = (uid, revision, self.digest)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        with self._compute:
            vector = self.fold_in(preferences, self.policy)
        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > 512:
                self._cache.popitem(last=False)
        return vector

    def fold_in(self, preferences, policy):
        liked = [self.rows[item] for item, value in preferences.items() if value == 1 and item in self.rows]
        if not liked or policy == 'popularity':
            return None
        if policy == 'centroid':
            vector = np.asarray(self.vectors[liked], dtype=np.float64).sum(axis=0)
        elif policy == 'als':
            a = self.gram.astype(np.float64).copy() + self.regularization * np.eye(self.dimension)
            b = np.zeros(self.dimension, dtype=np.float64)
            for item, value in preferences.items():
                if item not in self.rows:
                    continue
                y = np.asarray(self.raw[self.rows[item]], dtype=np.float64)
                confidence = 41 if value == 1 else 11
                a += (confidence - 1) * np.outer(y, y)
                if value == 1:
                    b += confidence * y
            chol = np.linalg.cholesky(a)
            vector = np.linalg.solve(chol.T, np.linalg.solve(chol, b))
        else:
            raise ValueError('Unknown policy')
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm < 1e-12:
            return None
        result = np.asarray(vector / norm, dtype=np.float32)
        result.flags.writeable = False
        return result
