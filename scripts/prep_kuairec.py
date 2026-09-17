#!/usr/bin/env python3
"""Convert official KuaiRec small/big matrix CSVs into RecServe's interaction schema.

Download from https://kuairec.com/ (CC BY-SA 4.0).
Small matrix: 1,411 users, 3,327 items, 4,676,570 interactions.
Big matrix: 7,176 users, 10,728 items, 12,530,806 interactions.

This is the QUALITY track. Do not use it as the QPS load.
"""
from __future__ import annotations

import argparse
import csv
import pathlib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="KuaiRec big_matrix.csv or small_matrix.csv")
    p.add_argument("--out", default="data/kuairec_interactions.csv")
    p.add_argument("--limit", type=int, default=0, help="optional row cap for smoke tests")
    args = p.parse_args()
    src = pathlib.Path(args.input)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(newline="", encoding="utf-8") as inf, out.open("w", newline="", encoding="utf-8") as ouf:
        reader = csv.DictReader(inf)
        w = csv.writer(ouf)
        w.writerow(["user", "item", "ts", "watch"])
        for row in reader:
            user = row.get("user_id") or row.get("user")
            item = row.get("video_id") or row.get("item_id") or row.get("item")
            ts = row.get("timestamp") or row.get("time") or row.get("ts") or "0"
            watch = row.get("watch_ratio") or row.get("watch") or "0"
            if user is None or item is None:
                continue
            w.writerow([user, item, ts, watch])
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} interactions -> {out}")
    print("KuaiRec counts if using the official files: small 1411/3327/4676570, big 7176/10728/12530806")


if __name__ == "__main__":
    main()
