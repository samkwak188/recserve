#!/usr/bin/env python3
"""Baselines for the real-data quality track, and a cross-check of the C++ path.

Two jobs, both about not fooling yourself:

  1. A recall number means nothing without something to compare it to. Random
     says what chance looks like; most-popular is the baseline every
     recommender must beat to have earned its existence. A model that ties
     popularity has learned popularity.

  2. The C++ service and this script compute recall@10 from the same
     embeddings by completely separate code paths -- one through
     recommend_sync, HNSW and the ranker, the other a numpy dot product. If
     they disagree, the serving path is wrong, and every latency number
     measured against it is measuring the wrong computation.

    python scripts/eval_baselines.py --prefix ml25m
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def exe(name: str) -> Path:
    for c in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
              ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    raise SystemExit(f"{name} not built")


def read_mat(path: Path, magic: int) -> np.ndarray:
    raw = path.read_bytes()
    m, n, d = struct.unpack_from("<Iii", raw, 0)
    if m != magic:
        raise SystemExit(f"bad magic in {path}: {m:#x}")
    return np.frombuffer(raw, np.float32, n * d, 12).reshape(n, d)


def read_pairs(path: Path) -> dict[int, set[int]]:
    out: dict[int, set[int]] = collections.defaultdict(set)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["query_row"])].add(int(r["item"]))
    return out


def score(topk_fn, gold: dict[int, set[int]], k: int) -> tuple[float, float]:
    rec, nd = [], []
    for u, g in gold.items():
        top = list(topk_fn(u))[:k]
        rec.append(len(set(top) & g) / len(g))
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top) if i in g)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(g))))
        nd.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(rec)), float(np.mean(nd))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ml25m")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--index", default="", help="skip to evaluate the C++ path exactly")
    ap.add_argument("--min-lift-over-popularity", type=float, default=0.0,
                    help="fail if ALS recall does not exceed popularity by this much")
    ap.add_argument("--max-cpp-delta", type=float, default=0.002)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = ROOT / "data"
    items = read_mat(d / f"{args.prefix}_catalog.bin", 0x43415431)
    users = read_mat(d / f"{args.prefix}_queries.bin", 0x51525931)
    gold = read_pairs(d / f"{args.prefix}_test.csv")
    seen = read_pairs(d / f"{args.prefix}_train.csv")
    mean_gold = float(np.mean([len(v) for v in gold.values()]))
    ceiling = min(args.k / mean_gold, 1.0)
    print(f"{items.shape[0]:,} items x {items.shape[1]} dims | {len(gold):,} eval users | "
          f"mean |gold| = {mean_gold:.1f} | recall@{args.k} ceiling = {ceiling:.3f}")

    pop = collections.Counter()
    for s in seen.values():
        for i in s:
            pop[i] += 1
    pop_rank = [i for i, _ in pop.most_common()]

    def by_random(u: int):
        return np.random.default_rng(u).choice(len(items), args.k, replace=False)

    def by_popularity(u: int):
        sn = seen.get(u, set())
        out = []
        for i in pop_rank:
            if i not in sn:
                out.append(i)
                if len(out) == args.k:
                    break
        return out

    def by_als(u: int):
        s = users[u] @ items.T
        sn = list(seen.get(u, ()))
        if sn:
            s[sn] = -np.inf
        idx = np.argpartition(-s, args.k)[: args.k]
        return idx[np.argsort(-s[idx])]

    rows = {}
    for name, fn in (("random", by_random), ("most_popular", by_popularity),
                     ("als_exact", by_als)):
        r, n = score(fn, gold, args.k)
        rows[name] = {"recall": r, "ndcg": n}
        print(f"  {name:14s} recall@{args.k}={r:.4f}  ndcg@{args.k}={n:.4f}")

    lift = rows["als_exact"]["recall"] - rows["most_popular"]["recall"]
    print(f"  ALS lift over popularity: {lift:+.4f} recall "
          f"({lift / max(rows['most_popular']['recall'], 1e-9) * 100:+.1f}%)")

    # Cross-check: the same embeddings, scored by the C++ service.
    cmd = [str(exe("recserve_eval")),
           "--catalog", f"data/{args.prefix}_catalog.bin",
           "--queries", f"data/{args.prefix}_queries.bin",
           "--test", f"data/{args.prefix}_test.csv",
           "--train", f"data/{args.prefix}_train.csv",
           "--k", str(args.k), "--json"]
    cmd += ["--index", args.index, "--ef", str(args.ef)] if args.index else ["--brute"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"recserve_eval failed:\n{out.stdout}\n{out.stderr}")
    cpp = json.loads(out.stdout.strip().splitlines()[-1])
    mode = f"HNSW ef={args.ef}" if args.index else "exact scan"
    print(f"  {'cpp_' + ('hnsw' if args.index else 'exact'):14s} "
          f"recall@{args.k}={cpp['recall_at_k']:.4f}  ndcg@{args.k}={cpp['ndcg_at_k']:.4f}  "
          f"retrieve_recall={cpp['retrieve_recall']:.4f}  p99={cpp['p99_us']:.0f}us  [{mode}]")

    payload = {"prefix": args.prefix, "k": args.k, "items": int(items.shape[0]),
               "dim": int(items.shape[1]), "eval_users": len(gold),
               "mean_gold": mean_gold, "recall_ceiling": ceiling,
               "baselines": rows, "als_lift_over_popularity": lift,
               "cpp": cpp, "cpp_mode": mode}
    if args.out:
        p = ROOT / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"-> {p}")

    fail = []
    if lift < args.min_lift_over_popularity:
        fail.append(f"ALS lift over popularity {lift:+.4f} < "
                    f"{args.min_lift_over_popularity:+.4f}")
    # Without an index the C++ path does an exact scan, so it must reproduce the
    # numpy result almost exactly; through HNSW a small approximation gap is
    # expected and is reported by retrieve_recall instead.
    if not args.index:
        delta = abs(cpp["recall_at_k"] - rows["als_exact"]["recall"])
        if delta > args.max_cpp_delta:
            fail.append(f"C++ exact recall differs from numpy by {delta:.4f} "
                        f"(limit {args.max_cpp_delta})")
        else:
            print(f"  cross-check OK: C++ and numpy agree to {delta:.5f}")
    if fail:
        for f in fail:
            print("FAIL " + f, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
