#!/usr/bin/env python3
"""Cross-validate RecServe's HNSW against hnswlib on identical data.

RecServe implements Malkov & Yashunin (arXiv:1603.09320) from the paper.
hnswlib (github.com/nmslib/hnswlib) is the authors' own implementation. Running
both over the same catalog, the same queries and matched (M, efConstruction, ef)
is the only way to tell an honest recall number from a broken graph.

Both are compared against exact float32 top-k computed with numpy, so "recall"
means the same thing on both sides.

    python scripts/validate_hnsw.py --items 65536 --dim 64 --clusters 1024

Requires: pip install hnswlib numpy
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import hnswlib
except ImportError:  # pragma: no cover
    raise SystemExit("pip install hnswlib")

ROOT = Path(__file__).resolve().parents[1]


def exe(name: str) -> Path:
    for cand in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
                 ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if cand.exists():
            return cand
    raise SystemExit(f"{name} not built")


def read_catalog(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, n, dim = struct.unpack_from("<Iii", raw, 0)
    if magic != 0x43415431:
        raise SystemExit(f"bad catalog magic {magic:#x}")
    return np.frombuffer(raw, dtype=np.float32, count=n * dim, offset=12).reshape(n, dim)


def read_queries(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, nq, dim = struct.unpack_from("<Iii", raw, 0)
    if magic != 0x51525931:
        raise SystemExit(f"bad query magic {magic:#x}")
    return np.frombuffer(raw, dtype=np.float32, count=nq * dim, offset=12).reshape(nq, dim)


def exact_topk(cat: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact inner-product top-k, chunked so a large catalog still fits in RAM."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for i in range(0, len(queries), 64):
        sims = queries[i : i + 64] @ cat.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
        out[i : i + 64] = np.take_along_axis(idx, order, axis=1)
    return out


def recall(pred: np.ndarray, gold: np.ndarray) -> float:
    return float(np.mean([len(set(p) & set(g)) / len(g) for p, g in zip(pred, gold)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=65536)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=1024)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=256)
    ap.add_argument("--ef", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--out", default="results/hnsw_validation.json")
    args = ap.parse_args()

    tmp = ROOT / "data"
    tmp.mkdir(exist_ok=True)
    cat_p, idx_p, qry_p = tmp / "_val_cat.bin", tmp / "_val_idx.bin", tmp / "_val_qry.bin"

    build = subprocess.run(
        [str(exe("recserve_fixture")), "--items", str(args.items), "--dim", str(args.dim),
         "--clusters", str(args.clusters), "--m", str(args.m),
         "--ef-construction", str(args.ef_construction), "--queries", str(args.queries),
         "--out-catalog", str(cat_p), "--out-index", str(idx_p), "--out-queries", str(qry_p)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if build.returncode != 0:
        raise SystemExit(f"fixture failed:\n{build.stdout}\n{build.stderr}")
    recserve_build = json.loads(build.stdout.strip().splitlines()[-1])

    cat = read_catalog(cat_p)
    queries = read_queries(qry_p)[: args.queries]
    print(f"catalog {cat.shape} queries {queries.shape}", file=sys.stderr)

    gold = exact_topk(cat, queries, args.k)

    # hnswlib with matched parameters. space="ip" matches RecServe's inner
    # product; hnswlib reports distance = 1 - ip, the ordering is the same.
    hl = hnswlib.Index(space="ip", dim=args.dim)
    hl.init_index(max_elements=len(cat), ef_construction=args.ef_construction, M=args.m)
    hl.add_items(cat, np.arange(len(cat)))

    rows = []
    for ef in args.ef:
        hl.set_ef(max(ef, args.k))
        hl_labels, _ = hl.knn_query(queries, k=args.k)
        hl_recall = recall(hl_labels, gold)

        rs = subprocess.run(
            [str(exe("recserve_bench")), "--load-catalog", str(cat_p), "--load-index", str(idx_p),
             "--mode", "simd", "--ef", str(ef), "--k", str(args.k), "--retrieve-k", str(args.k),
             "--n", "100", "--trials", "1", "--recall-probe", str(args.queries), "--json"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if rs.returncode != 0:
            raise SystemExit(f"bench failed:\n{rs.stdout}\n{rs.stderr}")
        rs_json = json.loads(rs.stdout.strip().splitlines()[-1])
        rs_recall = rs_json["retrieve_recall"]

        rows.append({"ef": ef, "hnswlib_recall": hl_recall, "recserve_recall": rs_recall,
                     "delta": rs_recall - hl_recall, "recserve_p99_us": rs_json["p99_mean_us"]})
        print(f"ef={ef:4d}  hnswlib={hl_recall:.4f}  recserve={rs_recall:.4f}  "
              f"delta={rs_recall - hl_recall:+.4f}")

    payload = {
        "items": args.items, "dim": args.dim, "clusters": args.clusters, "m": args.m,
        "ef_construction": args.ef_construction, "k": args.k, "queries": args.queries,
        "recserve_build": recserve_build, "rows": rows,
        "max_abs_delta": max(abs(r["delta"]) for r in rows),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nmax |delta| = {payload['max_abs_delta']:.4f}  ->  {out}")

    for p in (cat_p, idx_p, qry_p):
        try:
            os.remove(p)
        except OSError:
            pass
    # A correct implementation tracks the reference; 5 recall points of slack
    # covers RNG and the fact that RecServe searches layer 0 with one entry.
    return 0 if payload["max_abs_delta"] < 0.05 else 2


if __name__ == "__main__":
    raise SystemExit(main())
