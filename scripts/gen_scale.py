#!/usr/bin/env python3
"""Write a synthetic scale fixture. Default 100k x 64 f32 (~25MiB). Label as load fixture, not KuaiRec."""
from __future__ import annotations

import argparse
import pathlib
import struct
import random


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--out", default="data/scale.bin")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rng = random.Random(args.seed)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<II", args.n, args.dim))
        for _ in range(args.n):
            vec = [rng.gauss(0.0, 1.0) for _ in range(args.dim)]
            n = sum(x * x for x in vec) ** 0.5 or 1.0
            f.write(struct.pack("<" + "f" * args.dim, *[x / n for x in vec]))
    print(f"wrote {args.n} x {args.dim} -> {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
