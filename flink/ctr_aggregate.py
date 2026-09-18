#!/usr/bin/env python3
"""Flink job: per-item CTR aggregates from the interactions topic.

This is the offline/batch half of the feature the online C++ path computes
incrementally. Running both over identical records is how training/serving skew
gets measured instead of assumed (scripts/skew.py), and CI asserts this job's
output matches scripts/offline_ctr.py exactly on the same Kafka records.

It reads the same 17-byte record everything else reads:

    <Q event_time_ms   <I user_id   <I item_id   <B type

Design notes, because several obvious approaches do not actually run:

  - The topic carries raw bytes, not JSON or strings. PyFlink's KafkaSource has
    no Python-side deserializer for arbitrary binary, so this uses the Table API
    with `format = 'raw'`, which maps the whole payload to one BYTES column, and
    decodes the fields in Python UDFs.
  - `scan.bounded.mode = latest-offset` makes the source finite, so the job
    terminates and every window closes. An unbounded source with a one-hour
    window would emit nothing in a CI run and the job would hang.
  - Batch mode is used for the same reason: the filesystem sink commits its
    files on completion rather than on a checkpoint.

Run against the local Redpanda from docker-compose:

    pip install apache-flink==1.20.0
    curl -LO https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.3.0-1.20/flink-sql-connector-kafka-3.3.0-1.20.jar
    python scripts/kafka_produce.py --n 50000
    python flink/ctr_aggregate.py --jar file:///$PWD/flink-sql-connector-kafka-3.3.0-1.20.jar
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
from pathlib import Path

from pyflink.table import DataTypes, EnvironmentSettings, TableEnvironment
from pyflink.table.udf import udf

EVENT_BYTES = 17


@udf(result_type=DataTypes.BIGINT())
def ev_ms(payload: bytes) -> int:
    return struct.unpack_from("<Q", payload, 0)[0]


@udf(result_type=DataTypes.INT())
def ev_item(payload: bytes) -> int:
    return struct.unpack_from("<I", payload, 12)[0]


@udf(result_type=DataTypes.INT())
def ev_like(payload: bytes) -> int:
    return 1 if payload[16] == 1 else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brokers", default=os.environ.get("RECSERVE_BROKERS", "127.0.0.1:19092"))
    ap.add_argument("--topic", default="interactions")
    ap.add_argument("--group", default="recserve-flink-ctr")
    ap.add_argument("--window", default="1' HOUR", help="Flink interval literal tail")
    ap.add_argument("--jar", default="", help="file:// URL of flink-sql-connector-kafka")
    ap.add_argument("--out", default="data/offline/flink_ctr")
    ap.add_argument("--parallelism", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out).absolute()
    if out_dir.exists():
        for f in glob.glob(str(out_dir / "**" / "*"), recursive=True):
            if os.path.isfile(f):
                os.remove(f)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_env = TableEnvironment.create(EnvironmentSettings.in_batch_mode())
    t_env.get_config().set("parallelism.default", str(args.parallelism))
    if args.jar:
        t_env.get_config().set("pipeline.jars", args.jar)

    t_env.create_temporary_function("ev_ms", ev_ms)
    t_env.create_temporary_function("ev_item", ev_item)
    t_env.create_temporary_function("ev_like", ev_like)

    t_env.execute_sql(f"""
        CREATE TABLE interactions (
            payload BYTES,
            item AS ev_item(payload),
            is_like AS ev_like(payload),
            ts AS TO_TIMESTAMP_LTZ(ev_ms(payload), 3)
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{args.topic}',
            'properties.bootstrap.servers' = '{args.brokers}',
            'properties.group.id' = '{args.group}',
            'scan.startup.mode' = 'earliest-offset',
            'scan.bounded.mode' = 'latest-offset',
            'format' = 'raw'
        )
    """)

    t_env.execute_sql(f"""
        CREATE TABLE ctr_sink (
            item INT,
            window_start TIMESTAMP(3),
            views BIGINT,
            likes BIGINT,
            ctr DOUBLE
        ) WITH (
            'connector' = 'filesystem',
            'path' = '{out_dir.as_posix()}',
            'format' = 'csv'
        )
    """)

    # Event-time tumbling windows over the record's own timestamp, not arrival
    # order. The online path has no such luxury -- it applies events as they
    # arrive -- which is one of the two reasons the two sides can disagree; the
    # other is the publish boundary.
    result = t_env.execute_sql(f"""
        INSERT INTO ctr_sink
        SELECT
            item,
            CAST(window_start AS TIMESTAMP(3)) AS window_start,
            COUNT(*) AS views,
            SUM(CAST(is_like AS BIGINT)) AS likes,
            CAST(SUM(CAST(is_like AS BIGINT)) AS DOUBLE) / COUNT(*) AS ctr
        FROM TABLE(TUMBLE(TABLE interactions, DESCRIPTOR(ts), INTERVAL '{args.window}'))
        GROUP BY item, window_start, window_end
    """)
    result.wait()

    files = [f for f in glob.glob(str(out_dir / "**" / "*"), recursive=True)
             if os.path.isfile(f)]
    rows = sum(sum(1 for _ in open(f)) for f in files)
    print(f"flink wrote {rows} rows across {len(files)} file(s) -> {out_dir}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
