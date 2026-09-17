#!/usr/bin/env python3
"""Batch per-item CTR over an event log: the reference the Flink job must match.

flink/ctr_aggregate.py computes this on a cluster. This computes the identical
thing in plain Python so the offline/online skew number can be produced on any
host, including ones where apache-flink will not install (Windows/ARM64). CI
runs both and asserts they agree, which is what makes the Flink job trustworthy
rather than decorative.

Semantics, matched to the online path in include/recserve/nearline.hpp:
    views  = count of all events for the item
    likes  = count of events with type == 1
    ctr    = likes / views
over every event with event_time_ms <= --until (default: all of them).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

EVENT_BYTES = 17
EVENT_FMT = "<QIIB"
ROOT = Path(__file__).resolve().parents[1]


def read_events(path: Path, limit: int | None = None):
    raw = path.read_bytes()
    n = len(raw) // EVENT_BYTES
    if limit is not None:
        n = min(n, limit)
    for i in range(n):
        yield struct.unpack_from(EVENT_FMT, raw, i * EVENT_BYTES)


def aggregate(path: Path, until_ms: int | None, limit: int | None) -> dict[int, tuple[int, int]]:
    acc: dict[int, list[int]] = {}
    for event_ms, _user, item, etype in read_events(path, limit):
        if until_ms is not None and event_ms > until_ms:
            continue
        row = acc.setdefault(item, [0, 0])
        row[0] += 1
        row[1] += 1 if etype == 1 else 0
    return {k: (v[0], v[1]) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.bin")
    ap.add_argument("--until", type=int, default=None, help="max event_time_ms, inclusive")
    ap.add_argument("--limit", type=int, default=None, help="max records to read")
    ap.add_argument("--out", default="data/offline/offline_ctr.csv")
    args = ap.parse_args()

    acc = aggregate(ROOT / args.events, args.until, args.limit)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        f.write("item,views,likes,ctr\n")
        for item in sorted(acc):
            views, likes = acc[item]
            f.write(f"{item},{views},{likes},{(likes / views) if views else 0.0:.9f}\n")
    print(f"items={len(acc)} events={sum(v for v, _ in acc.values())} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
