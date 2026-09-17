#!/usr/bin/env python3
"""Measure online/offline feature skew.

Training reads the offline aggregate. Serving reads the online snapshot. When
they disagree, a model is trained on features the serving path will never
produce -- the failure that makes an offline AUC win evaporate online.

This measures the disagreement instead of hoping there is none:

  online   recserve_nearline --replay, which publishes on an interval and
           deliberately does NOT flush, so whatever is still staged at the end
           is invisible to queries exactly as it would be on a live host
  offline  the same records aggregated in full up to the same watermark

Everything after the last publish boundary is the skew. It is a function of the
publish interval, so the script sweeps it and reports the curve.

    python scripts/skew.py --events data/events.bin --n 200000
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_ctr import aggregate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def exe(name: str) -> Path:
    for c in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
              ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    raise SystemExit(f"{name} not built")


def read_csv(path: Path) -> dict[int, tuple[int, int, float]]:
    out: dict[int, tuple[int, int, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["item"])] = (int(row["views"]), int(row["likes"]), float(row["ctr"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.bin")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--items", type=int, default=16384)
    ap.add_argument("--publish-ms", type=int, nargs="+", default=[100, 1000, 5000, 30000])
    ap.add_argument("--out", default="results/skew.json")
    args = ap.parse_args()

    rows = []
    print(f"{'publish_ms':>10s} {'unpublished':>12s} {'items_differ':>13s} {'max_dctr':>9s} "
          f"{'mean_dctr':>10s} {'views_missing':>14s}")
    for pub in args.publish_ms:
        dump = ROOT / "data" / "offline" / f"online_ctr_{pub}.csv"
        res = subprocess.run(
            [str(exe("recserve_nearline")), "--replay", str(args.n), "--publish-ms", str(pub),
             "--items", str(args.items), "--events", args.events, "--dump-features", str(dump)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise SystemExit(f"replay failed:\n{res.stdout}\n{res.stderr}")
        meta = json.loads(res.stdout.strip().splitlines()[-1])

        online = read_csv(dump)
        # The batch job sees every record the consumer read, including the ones
        # still staged behind the publish boundary. Cutting offline at the
        # PUBLISHED watermark instead would truncate both sides identically and
        # report zero skew by construction.
        offline = aggregate(ROOT / args.events, None, args.n)

        differ = 0
        max_d = 0.0
        sum_d = 0.0
        views_missing = 0
        for item, (v_off, l_off) in offline.items():
            v_on, _l_on, ctr_on = online.get(item, (0, 0, 0.0))
            ctr_off = (l_off / v_off) if v_off else 0.0
            d = abs(ctr_on - ctr_off)
            if v_on != v_off:
                differ += 1
                views_missing += v_off - v_on
            max_d = max(max_d, d)
            sum_d += d
        mean_d = sum_d / len(offline) if offline else 0.0

        cell = {
            "publish_ms": pub,
            "consumed": meta["consumed"],
            "published": meta["published"],
            "unpublished": meta["unpublished"],
            "publishes": meta["publishes"],
            "watermark_ms": meta["watermark_ms"],
            "items_offline": len(offline),
            "items_differ": differ,
            "views_missing": views_missing,
            "max_abs_ctr_delta": max_d,
            "mean_abs_ctr_delta": mean_d,
        }
        rows.append(cell)
        print(f"{pub:10d} {cell['unpublished']:12d} {differ:13d} {max_d:9.5f} {mean_d:10.6f} "
              f"{views_missing:14d}")

    payload = {"events": args.events, "n": args.n, "items": args.items, "rows": rows}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
