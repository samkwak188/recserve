#!/usr/bin/env python3
"""A/B the mutex feature store against the RCU snapshot under live ingest.

Same catalog, same serving threads, same event rate; the only difference is how
features are read. Repeats each cell and reports the median so one scheduling
hiccup does not decide the result.

Also sweeps the publish interval, because the snapshot's advantage is not free:
a longer interval means a better tail and staler features. That curve is the
actual engineering decision.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exe(name: str) -> Path:
    for c in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
              ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    raise SystemExit(f"{name} not built")


def run(store: str, rate: int, threads: int, seconds: int, publish_ms: int, items: int) -> dict:
    out = subprocess.check_output(
        [str(exe("recserve_nearline")), "--feature-store", store, "--rate", str(rate),
         "--serve-threads", str(threads), "--seconds", str(seconds),
         "--publish-ms", str(publish_ms), "--items", str(items), "--json"],
        cwd=ROOT, text=True,
    )
    return json.loads(out.strip().splitlines()[-1])


def med(rows: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=int, nargs="+", default=[10_000, 100_000, 500_000])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=4)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--items", type=int, default=16384)
    ap.add_argument("--publish-sweep", type=int, nargs="+", default=[10, 50, 100, 500])
    ap.add_argument("--out", default="results/nearline_ab.json")
    args = ap.parse_args()

    ab = []
    print(f"{'store':9s} {'rate':>8s} {'qps':>9s} {'p50':>7s} {'p99':>8s} {'p99.9':>9s} "
          f"{'max':>9s} {'fresh_p99':>10s}")
    for rate in args.rates:
        for store in ("mutex", "snapshot"):
            rows = [run(store, rate, args.threads, args.seconds, 100, args.items)
                    for _ in range(args.trials)]
            cell = {
                "store": store, "rate": rate, "trials": args.trials,
                "serve_qps": med(rows, "serve_qps"),
                "p50_us": med(rows, "serve_p50_us"),
                "p99_us": med(rows, "serve_p99_us"),
                "p999_us": med(rows, "serve_p999_us"),
                "max_us": med(rows, "serve_max_us"),
                "ingest_eps": med(rows, "ingest_eps"),
                "freshness_p99_ms": med(rows, "freshness_p99_ms"),
                "p99_spread_us": [r["serve_p99_us"] for r in rows],
            }
            ab.append(cell)
            print(f"{store:9s} {rate:8d} {cell['serve_qps']:9.0f} {cell['p50_us']:7.1f} "
                  f"{cell['p99_us']:8.1f} {cell['p999_us']:9.1f} {cell['max_us']:9.1f} "
                  f"{cell['freshness_p99_ms']:10.1f}")

    print(f"\n{'publish_ms':>10s} {'p99':>8s} {'p99.9':>9s} {'max':>9s} {'fresh_p50':>10s} "
          f"{'fresh_p99':>10s} {'publish_p99_us':>15s}")
    sweep = []
    hi = max(args.rates)
    for pub in args.publish_sweep:
        rows = [run("snapshot", hi, args.threads, args.seconds, pub, args.items)
                for _ in range(args.trials)]
        cell = {
            "publish_ms": pub,
            "p99_us": med(rows, "serve_p99_us"),
            "p999_us": med(rows, "serve_p999_us"),
            "max_us": med(rows, "serve_max_us"),
            "freshness_p50_ms": med(rows, "freshness_p50_ms"),
            "freshness_p99_ms": med(rows, "freshness_p99_ms"),
            "publish_p99_us": med(rows, "publish_p99_us"),
            "publishes": med(rows, "publishes"),
        }
        sweep.append(cell)
        print(f"{pub:10d} {cell['p99_us']:8.1f} {cell['p999_us']:9.1f} {cell['max_us']:9.1f} "
              f"{cell['freshness_p50_ms']:10.1f} {cell['freshness_p99_ms']:10.1f} "
              f"{cell['publish_p99_us']:15.1f}")

    payload = {"config": vars(args), "ab": ab, "publish_sweep": sweep}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
