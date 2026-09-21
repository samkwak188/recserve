#!/usr/bin/env python3
"""Global-time policy selection through real RSV2 and shared serving filters."""
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prep_movielens import fetch, load_ratings
from scripts.model_v2 import write_bundle
from scripts.production_runner import source_identity
from recserve_app.model import Model, file_hash
from recserve_app.retrieval import Retrieval
from recserve_app.selection import select_items


def boundaries(raw):
    stamps = np.sort(raw[:, 3].astype(np.int64))
    return int(stamps[int(.8 * len(stamps))]), int(stamps[int(.9 * len(stamps))])


def examples(raw, lower, upper, available, trained_users, limit):
    future = raw[(raw[:, 3] > lower) & (raw[:, 3] <= upper) & (raw[:, 2] >= 4)]
    candidates = np.unique(future[:, 0].astype(int))
    rng = np.random.default_rng(13)
    if len(candidates) > limit:
        candidates = np.sort(rng.choice(candidates, limit, replace=False))
    # Group once rather than scan 25 million rows for each evaluated user.
    selected = raw[np.isin(raw[:, 0], candidates)]
    selected = selected[np.lexsort((selected[:, 3], selected[:, 0]))]
    groups = np.split(selected, np.flatnonzero(np.diff(selected[:, 0])) + 1)
    result, omitted = [], Counter()
    for rows in groups:
        user = int(rows[0, 0])
        cohort = 'warm' if user in trained_users else 'held_out_user'
        history = rows[rows[:, 3] <= lower]
        target = rows[(rows[:, 3] > lower) & (rows[:, 3] <= upper) & (rows[:, 2] >= 4)]
        if cohort == 'held_out_user':
            positives = history[(history[:, 2] >= 4) & np.isin(history[:, 1], list(available))]
            if len(positives) >= 5:
                history = positives[-5:]
            else:
                seeds = target[np.isin(target[:, 1], list(available))][:5]
                if len(seeds) < 5:
                    omitted['insufficient_onboarding'] += 1
                    continue
                history = seeds
                target = target[target[:, 3] > seeds[-1, 3]]
        history = history[(history[:, 2] >= 4) | (history[:, 2] <= 2)][-500:]
        prefs = {int(row[1]): 1 if row[2] >= 4 else -1 for row in history if int(row[1]) in available}
        all_gold = {int(row[1]) for row in target if int(row[1]) not in prefs}
        omitted['unsupported_target_items'] += len(all_gold - available)
        gold = all_gold & available
        if not gold:
            omitted['no_supported_targets'] += 1
            continue
        result.append((user, cohort, prefs, gold))
    return result, dict(omitted)


def evaluate(model, retrieval, rows, policy):
    scores, recall, coverage, head, cohorts = [], [], set(), [], {}
    popular = set(model.popularity[:max(1, len(model.popularity) // 10)])
    for user, cohort, prefs, gold in rows:
        result = select_items(model, retrieval, model.fold_in(prefs, policy), set(prefs), 10, policy)
        if result['degraded']:
            raise RuntimeError('Retrieval failure invalidates quality evidence')
        ids = [item['id'] for item in result['items']]
        gains = [1 / np.log2(rank + 2) if item in gold else 0 for rank, item in enumerate(ids)]
        ndcg = sum(gains) / sum(1 / np.log2(rank + 2) for rank in range(min(10, len(gold))))
        scores.append(float(ndcg))
        recall.append(len(set(ids) & gold) / len(gold))
        coverage.update(ids)
        head.append(sum(item in popular for item in ids) / max(1, len(ids)))
        cohorts.setdefault(cohort, []).append(float(ndcg))
    if not scores:
        raise RuntimeError('No evaluable users')
    return dict(users=len(scores), ndcg10=float(np.mean(scores)), recall10=float(np.mean(recall)),
        catalog_coverage=len(coverage) / len(model.movies), popular_decile_share=float(np.mean(head)),
        cohorts={name: {'users': len(values), 'ndcg10': float(np.mean(values))} for name, values in cohorts.items()},
        user_ndcg=scores)


def confidence(candidate, baseline):
    diff = np.asarray(candidate) - np.asarray(baseline)
    rng = np.random.default_rng(13)
    samples = np.mean(diff[rng.integers(0, len(diff), size=(1000, len(diff)))], axis=1)
    return list(map(float, np.quantile(samples, [.025, .975])))


def free_port():
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        return connection.getsockname()[1]


def main():
    from scipy.sparse import csr_matrix
    from implicit.als import AlternatingLeastSquares
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['small', '25m'], default='small')
    parser.add_argument('--max-eval-users', type=int, default=1024)
    args = parser.parse_args()
    archive = fetch(args.dataset, ROOT / 'data/movielens')
    raw = load_ratings(archive)
    cut_train, cut_validation = boundaries(raw)
    train = raw[(raw[:, 3] <= cut_train) & (raw[:, 2] >= 4) & (raw[:, 0].astype(int) % 5 != 0)]
    item_ids, columns = np.unique(train[:, 1].astype(int), return_inverse=True)
    user_ids, indices = np.unique(train[:, 0].astype(int), return_inverse=True)
    matrix = csr_matrix((np.full(len(train), 41, dtype=np.float32), (indices, columns)),
                        shape=(len(user_ids), len(item_ids)))
    als = AlternatingLeastSquares(factors=32, regularization=.05, iterations=15, random_state=13,
                                  use_gpu=False, num_threads=1)
    als.fit(matrix, show_progress=False)
    with zipfile.ZipFile(archive) as archive_file:
        movie_path = next(name for name in archive_file.namelist() if name.endswith('/movies.csv'))
        with archive_file.open(movie_path) as stream:
            metadata = {int(row['movieId']): row for row in csv.DictReader(io.TextIOWrapper(stream, 'utf-8'))}
        license_path = next(name for name in archive_file.namelist() if name.endswith('/README.txt'))
        license_text = archive_file.read(license_path).decode()
    movies = []
    for item in item_ids:
        entry = metadata[int(item)]
        match = re.search(r'\((\d{4})\)$', entry['title'])
        movies.append(dict(id=int(item), title=entry['title'], year=int(match[1]) if match else None,
                           genres=entry['genres'].split('|'), available=True))
    counts = np.bincount(columns, minlength=len(item_ids))
    popularity = [int(item_ids[row]) for row in np.lexsort((item_ids, -counts))]
    training = dict(dataset=args.dataset, dataset_sha256=file_hash(archive), train_end=cut_train,
        validation_end=cut_validation, seed=13, factors=32, iterations=15,
        held_out_user_rule='raw user ID modulo 5 equals 0', train_users=len(user_ids), train_events=len(train),
        source=source_identity(ROOT))
    output = ROOT / '.cache/production/quality' / (args.dataset + '-' + str(time.time_ns()))
    output.mkdir(parents=True)
    manifest = write_bundle(output / 'candidate', als.item_factors, movies, popularity, 'als', training,
                            license_text, ROOT / 'build-production/recserve_fixture')
    model = Model(manifest)
    port, metrics = free_port(), free_port()
    while port == metrics:
        metrics = free_port()
    valid, valid_omitted = examples(raw, cut_train, cut_validation, set(model.rows), set(map(int, user_ids)), args.max_eval_users)
    test, test_omitted = examples(raw, cut_validation, int(raw[:, 3].max()), set(model.rows), set(map(int, user_ids)), args.max_eval_users)
    with (output / 'server.log').open('w') as log:
        process = subprocess.Popen([sys.executable, 'scripts/model_v2.py', 'serve', '--manifest', str(manifest),
            '--port', str(port), '--metrics-port', str(metrics)], cwd=ROOT, stdout=log, stderr=log)
        upstream = Retrieval('127.0.0.1', port, model.digest, model.dimension)
        try:
            for _ in range(200):
                try:
                    upstream.ready()
                    break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError('Retrieval startup failed')
                    time.sleep(.05)
            else:
                raise RuntimeError('Retrieval startup timeout')
            validation = {policy: evaluate(model, upstream, valid, policy) for policy in ('popularity', 'centroid', 'als')}
            candidate = max(validation, key=lambda policy: validation[policy]['ndcg10'])
            baseline = evaluate(model, upstream, test, 'popularity')
            candidate_test = baseline if candidate == 'popularity' else evaluate(model, upstream, test, candidate)
            interval = confidence(candidate_test['user_ndcg'], baseline['user_ndcg'])
            selected = candidate if candidate != 'popularity' and interval[0] > 0 else 'popularity'
            probes = []
            for _, _, prefs, _ in valid[:128]:
                vector = model.fold_in(prefs, 'als')
                if vector is not None:
                    exact = set(np.argsort(-(model.vectors @ vector))[:128])
                    got = {row for row, _ in upstream.query(vector, 128)}
                    probes.append(len(exact & got) / len(exact))
            if not probes or np.mean(probes) < .98:
                raise RuntimeError('ANN recall@128 gate failed')
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    # Create a new immutable release directory; never alter a running model basis.
    release = write_bundle(output / 'release', als.item_factors, movies, popularity, selected, training,
                           license_text, ROOT / 'build-production/recserve_fixture')
    report = dict(training=training, validation=validation, test={'popularity': baseline, 'candidate': candidate_test},
        validation_candidate=candidate, paired_ndcg95=interval, bootstrap_resamples=1000, selected_policy=selected,
        ann_recall128=float(np.mean(probes)), omitted={'validation': valid_omitted, 'test': test_omitted},
        release_manifest=str(release), release_sha256=file_hash(release),
        scope='development snapshot; no online lift' if args.dataset == 'small' else 'local benchmark; no redistribution permission inferred')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(output / 'report.json'), 'selected_policy': selected,
                      'paired_ndcg95': interval, 'ann_recall128': report['ann_recall128']}))


if __name__ == '__main__':
    main()
