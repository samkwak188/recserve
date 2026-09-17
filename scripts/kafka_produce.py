#!/usr/bin/env python3
"""Produce RecServe EventRecords to a Kafka/Redpanda topic.

Requires: pip install kafka-python
Broker defaults to docker-compose Redpanda on localhost:19092.

The C++ nearline job still owns window/CTR/freshness logic. This script only
carries the same 17-byte records. Enable C++ librdkafka with
-DRECSERVE_WITH_RDKAFKA=ON when the library is installed; otherwise replay
data/events.bin.
"""
from __future__ import annotations

import argparse
import struct
import time

try:
    from kafka import KafkaProducer
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install kafka-python") from e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default="127.0.0.1:19092")
    p.add_argument("--topic", default="interactions")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    prod = KafkaProducer(bootstrap_servers=args.broker)
    t0 = int(time.time() * 1000)
    for i in range(args.n):
        rec = struct.pack("<QIIB", t0 + i, i % 64, i % 256, 1 if i % 7 == 0 else 0)
        prod.send(args.topic, rec)
    prod.flush()
    print(f"sent {args.n} records to {args.topic}@{args.broker}")


if __name__ == "__main__":
    main()
