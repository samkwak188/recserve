#!/usr/bin/env python3
"""Train real item embeddings on MovieLens and export them as a RecServe catalog.

Everything else in this repo runs on synthetic vectors. That is fine for
measuring kernels and layouts -- a dot product does not care where the numbers
came from -- but it makes recall@10 a statement about high-dimensional geometry
rather than about recommendation quality. i.i.d. Gaussian directions have no
intrinsic structure, which is the documented worst case for graph ANN, so the
published recall numbers look far worse than a real embedding table would give.

This fixes that end of the project:

  1. download MovieLens (ml-latest-small for CI, ml-25m for the real board)
  2. split per user by timestamp -- the last `--test-frac` of each user's
     history is held out, so the model never sees the future it is scored on
  3. fit implicit-feedback ALS (Hu, Koren & Volinsky, "Collaborative Filtering
     for Implicit Feedback Datasets", ICDM 2008) on the training half
  4. write item factors as a RecServe catalog, user factors as the query set,
     and the held-out interactions as ground truth

Ratings are treated as implicit signal: a rating >= --min-rating is a positive
interaction with confidence 1 + alpha * rating. Predicting the rating value is
a different task from ranking a catalog, and ranking is what is being served.

Vectors are L2-normalised on the way out, because the engine scores with inner
product and the index is built for it; normalising makes inner product and
cosine agree, which is the usual choice for retrieval.

    python scripts/prep_movielens.py --dataset small
    python scripts/prep_movielens.py --dataset 25m --factors 64
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def fetch(dataset: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / f"ml-{dataset}.zip"
    if not zpath.exists():
        url = URLS[dataset]
        print(f"downloading {url}", file=sys.stderr)
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=600, context=ctx) as r:
            zpath.write_bytes(r.read())
    return zpath


def load_ratings(zpath: Path) -> np.ndarray:
    """Returns an (n, 4) array of (user, item, rating, ts).

    ml-25m is 25 million rows; numpy's text parsers take minutes on that, so
    pandas does the read when it is available (it ships with implicit anyway).
    """
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith("ratings.csv"))
        print(f"reading {name}", file=sys.stderr)
        try:
            import pandas as pd
            with z.open(name) as fh:
                df = pd.read_csv(fh, dtype={"userId": np.int64, "movieId": np.int64,
                                            "rating": np.float32, "timestamp": np.int64})
            return np.column_stack([df["userId"].to_numpy(), df["movieId"].to_numpy(),
                                    df["rating"].to_numpy(), df["timestamp"].to_numpy()])
        except ImportError:
            with z.open(name) as fh:
                return np.genfromtxt(io.TextIOWrapper(fh, "utf-8"), delimiter=",",
                                     skip_header=1, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["small", "25m"], default="small")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--regularization", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--min-rating", type=float, default=4.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-user-events", type=int, default=10)
    ap.add_argument("--max-eval-users", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cache", default="data/movielens")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    from scipy.sparse import csr_matrix
    from implicit.als import AlternatingLeastSquares

    prefix = args.prefix or f"ml{args.dataset}"
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_ratings(fetch(args.dataset, ROOT / args.cache))
    users_raw = raw[:, 0].astype(np.int64)
    items_raw = raw[:, 1].astype(np.int64)
    ratings = raw[:, 2].astype(np.float32)
    stamps = raw[:, 3].astype(np.int64)

    keep = ratings >= args.min_rating
    users_raw, items_raw, ratings, stamps = (users_raw[keep], items_raw[keep],
                                             ratings[keep], stamps[keep])
    print(f"{len(ratings):,} positive interactions (rating >= {args.min_rating})",
          file=sys.stderr)

    # Contiguous ids; the C++ side indexes catalogs and query sets by position.
    uniq_u, u_idx = np.unique(users_raw, return_inverse=True)
    uniq_i, i_idx = np.unique(items_raw, return_inverse=True)
    n_users, n_items = len(uniq_u), len(uniq_i)
    print(f"{n_users:,} users x {n_items:,} items", file=sys.stderr)

    # Per-user temporal split: sort by (user, ts) and hold out each user's tail.
    order = np.lexsort((stamps, u_idx))
    u_sorted, i_sorted = u_idx[order], i_idx[order]
    counts = np.bincount(u_sorted, minlength=n_users)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rank_in_user = np.arange(len(u_sorted)) - starts[u_sorted]
    n_test_per_user = np.maximum((counts * args.test_frac).astype(np.int64), 1)
    n_train_per_user = counts - n_test_per_user
    is_test = rank_in_user >= n_train_per_user[u_sorted]
    # Users with too little history are train-only: a one-event user tells you
    # nothing about ranking quality.
    too_small = counts < args.min_user_events
    is_test &= ~too_small[u_sorted]

    tr_u, tr_i = u_sorted[~is_test], i_sorted[~is_test]
    te_u, te_i = u_sorted[is_test], i_sorted[is_test]
    print(f"train {len(tr_u):,} / test {len(te_u):,} interactions", file=sys.stderr)

    conf = 1.0 + args.alpha
    train = csr_matrix((np.full(len(tr_u), conf, dtype=np.float32), (tr_u, tr_i)),
                       shape=(n_users, n_items))

    model = AlternatingLeastSquares(factors=args.factors, regularization=args.regularization,
                                    iterations=args.iterations, random_state=args.seed,
                                    use_gpu=False)
    print(f"fitting ALS: {args.factors} factors, {args.iterations} iterations",
          file=sys.stderr)
    model.fit(train)

    item_f = np.asarray(model.item_factors, dtype=np.float32)
    user_f = np.asarray(model.user_factors, dtype=np.float32)

    def l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        return (x / n).astype(np.float32)

    item_f, user_f = l2(item_f), l2(user_f)

    # Evaluate only users that actually have held-out items.
    eval_users = np.unique(te_u)
    rng = np.random.default_rng(args.seed)
    if len(eval_users) > args.max_eval_users:
        eval_users = np.sort(rng.choice(eval_users, args.max_eval_users, replace=False))
    remap = {int(u): k for k, u in enumerate(eval_users)}
    mask = np.isin(te_u, eval_users)
    te_u_eval, te_i_eval = te_u[mask], te_i[mask]

    cat_p = out / f"{prefix}_catalog.bin"
    qry_p = out / f"{prefix}_queries.bin"
    test_p = out / f"{prefix}_test.csv"
    seen_p = out / f"{prefix}_train.csv"

    with cat_p.open("wb") as f:
        f.write(struct.pack("<Iii", 0x43415431, n_items, args.factors))
        f.write(item_f.tobytes())
    with qry_p.open("wb") as f:
        f.write(struct.pack("<Iii", 0x51525931, len(eval_users), args.factors))
        f.write(user_f[eval_users].tobytes())
    with test_p.open("w", newline="") as f:
        f.write("query_row,item\n")
        for u, i in zip(te_u_eval, te_i_eval):
            f.write(f"{remap[int(u)]},{int(i)}\n")
    # The items a user already interacted with in train, so evaluation can
    # exclude them the way a real ranker would.
    tr_mask = np.isin(tr_u, eval_users)
    with seen_p.open("w", newline="") as f:
        f.write("query_row,item\n")
        for u, i in zip(tr_u[tr_mask], tr_i[tr_mask]):
            f.write(f"{remap[int(u)]},{int(i)}\n")

    meta = {
        "dataset": args.dataset, "factors": args.factors, "iterations": args.iterations,
        "regularization": args.regularization, "alpha": args.alpha,
        "min_rating": args.min_rating, "test_frac": args.test_frac,
        "n_users": int(n_users), "n_items": int(n_items),
        "train_interactions": int(len(tr_u)), "test_interactions": int(len(te_u)),
        "eval_users": int(len(eval_users)),
        "eval_test_interactions": int(len(te_i_eval)),
        "catalog": cat_p.name, "queries": qry_p.name, "test": test_p.name,
        "train_csv": seen_p.name,
    }
    (out / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
