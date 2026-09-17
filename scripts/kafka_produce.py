#!/usr/bin/env python3
"""Produce RecServe EventRecords to a Kafka/Redpanda topic at a controlled rate.

Same 17-byte record the C++ consumer, the file replay and the Flink job read:

    <Q event_time_ms   <I user_id   <I item_id   <B type

Two things this does that matter for the numbers downstream:

  --rate   paces production so consumer lag and freshness mean something. A
           flat-out producer just measures how fast the loop runs.
  --zipf   draws items from a Zipf popularity distribution, matching
           include/recserve/nearline.hpp. Uniform items would make every
           per-item counter equally warm and hide that the skew budget is spent
           almost entirely on the cold tail.

Timestamps are real wall-clock milliseconds, so freshness measured by the
consumer includes the broker round trip.

Requires: pip install kafka-python
"""
from __future__ import annotations

import argparse
import random
import struct
import sys
import time

try:
    from kafka import KafkaProducer
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install kafka-python") from e

EVENT_FMT = "<QIIB"


def zipf_cdf(n: int, s: float) -> list[float]:
    acc = 0.0
    cdf = []
    for i in range(n):
        acc += 1.0 / ((i + 1) ** s)
        cdf.append(acc)
    return [c / acc for c in cdf]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default="127.0.0.1:19092")
    p.add_argument("--topic", default="interactions")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--users", type=int, default=4096)
    p.add_argument("--items", type=int, default=16384)
    p.add_argument("--rate", type=int, default=20_000, help="events/s; 0 = unthrottled")
    p.add_argument("--zipf", type=float, default=1.0, help="0 = uniform")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    cdf = zipf_cdf(args.items, args.zipf) if args.zipf > 0 else None
    prod = KafkaProducer(bootstrap_servers=args.broker, linger_ms=5, acks=1)

    import bisect

    t0 = time.perf_counter()
    for i in range(args.n):
        if cdf is None:
            item = rng.randrange(args.items)
        else:
            item = min(bisect.bisect_left(cdf, rng.random()), args.items - 1)
        rec = struct.pack(
            EVENT_FMT,
            int(time.time() * 1000),
            rng.randrange(args.users),
            item,
            1 if rng.randrange(7) == 0 else 0,
        )
        prod.send(args.topic, rec)

        if args.rate > 0 and (i & 511) == 0:
            target = i / args.rate
            drift = target - (time.perf_counter() - t0)
            if drift > 0:
                time.sleep(drift)

    prod.flush()
    elapsed = time.perf_counter() - t0
    print(f"sent {args.n} records to {args.topic}@{args.broker} "
          f"in {elapsed:.2f}s ({args.n / elapsed:.0f} eps)", file=sys.stderr)


if __name__ == "__main__":
    main()
