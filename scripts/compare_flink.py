#!/usr/bin/env python3
"""Assert the Flink job and the plain-Python reference agree, record for record.

flink/ctr_aggregate.py runs event-time tumbling windows on a Flink cluster over
the Kafka topic. scripts/offline_ctr.py computes the same per-item view/like
counts in a loop. They share no code.

If they disagree, one of them is wrong and every skew number downstream is
meaningless. That is why this is an equality check with zero tolerance rather
than a report: the two sides consume byte-identical records (the producer dumps
exactly what it sent), so any difference is a bug, not sampling.

Flink emits one row per (item, window); this sums across windows before
comparing, which is correct regardless of where the window boundaries land.

    python scripts/compare_flink.py --flink data/offline/flink_ctr \\
                                    --reference data/offline/offline_ctr.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_flink(path: Path) -> dict[int, tuple[int, int]]:
    """Sum (views, likes) per item across every window and part file."""
    acc: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    files = [f for f in glob.glob(str(path / "**" / "*"), recursive=True)
             if os.path.isfile(f) and not os.path.basename(f).startswith(".")]
    if not files:
        raise SystemExit(f"no Flink output under {path}")
    rows = 0
    for f in files:
        with open(f, newline="") as fh:
            # The filesystem connector writes headerless CSV:
            # item, window_start, views, likes, ctr
            for row in csv.reader(fh):
                if not row:
                    continue
                item, _ws, views, likes = int(row[0]), row[1], int(row[2]), int(row[3])
                acc[item][0] += views
                acc[item][1] += likes
                rows += 1
    print(f"flink: {rows} window rows over {len(acc)} items in {len(files)} file(s)")
    return {k: (v[0], v[1]) for k, v in acc.items()}


def read_reference(path: Path) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["item"])] = (int(r["views"]), int(r["likes"]))
    print(f"reference: {len(out)} items")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flink", default="data/offline/flink_ctr")
    ap.add_argument("--reference", default="data/offline/offline_ctr.csv")
    args = ap.parse_args()

    flink = read_flink(ROOT / args.flink)
    ref = read_reference(ROOT / args.reference)

    only_flink = sorted(set(flink) - set(ref))
    only_ref = sorted(set(ref) - set(flink))
    mismatched = sorted(i for i in set(flink) & set(ref) if flink[i] != ref[i])

    fl_views = sum(v for v, _ in flink.values())
    rf_views = sum(v for v, _ in ref.values())
    fl_likes = sum(l for _, l in flink.values())
    rf_likes = sum(l for _, l in ref.values())
    print(f"totals  flink views={fl_views} likes={fl_likes} | "
          f"reference views={rf_views} likes={rf_likes}")

    if only_flink or only_ref or mismatched:
        print(f"FAIL items only in flink={len(only_flink)} only in reference={len(only_ref)} "
              f"mismatched={len(mismatched)}", file=sys.stderr)
        for i in mismatched[:10]:
            print(f"  item {i}: flink={flink[i]} reference={ref[i]}", file=sys.stderr)
        for i in only_flink[:5]:
            print(f"  item {i}: flink={flink[i]} reference=absent", file=sys.stderr)
        for i in only_ref[:5]:
            print(f"  item {i}: flink=absent reference={ref[i]}", file=sys.stderr)
        return 2

    print(f"PASS  {len(ref)} items agree exactly, {rf_views} events accounted for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
