#!/usr/bin/env python3
"""Tiny perf CI gate: fail if p99 regresses more than --max-ratio vs a stored baseline."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import math


def parse_p99(text: str) -> float:
    m = re.search(r"p99_mean_us=([0-9.]+)", text)
    if not m:
        raise SystemExit("could not parse p99_mean_us from bench output:\n" + text)
    return float(m.group(1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, help="path to recserve_bench")
    p.add_argument("--baseline", default="data/perf_baseline.json")
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--max-ratio", type=float, default=1.25)
    p.add_argument("--items", type=int, default=1024)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    cmd = [args.bench, "--items", str(args.items), "--n", str(args.n), "--mode", "simd", "--trials", "3"]
    out = subprocess.check_output(cmd, text=True)
    p99 = parse_p99(out)
    if not math.isfinite(p99) or p99 <= 0:
        raise SystemExit("invalid p99 measurement")
    path = pathlib.Path(args.baseline)
    if args.write_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"p99_mean_us": p99, "items": args.items, "n": args.n}, indent=2) + "\n")
        print(f"wrote baseline {p99:.3f} us -> {path}")
        return 0
    if not path.exists():
        raise SystemExit("baseline missing; create explicitly with --write-baseline")
    base = json.loads(path.read_text())["p99_mean_us"]
    print(f"baseline_us={base:.3f} current_us={p99:.3f} ratio={p99 / base:.3f}")
    if p99 > base * args.max_ratio:
        print("perf CI failed: p99 regression", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
