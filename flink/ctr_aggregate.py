#!/usr/bin/env python3
"""Flink job: per-item CTR aggregates from the interactions topic.

This is the offline/batch half of the same feature the online path computes
incrementally. Running both over identical records is how training/serving skew
gets measured instead of assumed (see scripts/skew.py).

Reads the same 17-byte record the C++ consumer reads:
    <Q event_time_ms  <I user_id  <I item_id  <B type

Windowing is event-time tumbling with a bounded-out-of-orderness watermark, so
a late event lands in the window it belongs to rather than the one it arrived
in. The online path has no such luxury: it applies events in arrival order,
which is one of the two reasons the two sides can disagree (the other being the
publish boundary).

Run inside the docker-compose stack:

    docker compose up -d redpanda flink-jobmanager flink-taskmanager
    python scripts/kafka_produce.py --n 200000
    docker compose exec flink-jobmanager flink run -py /opt/recserve/flink/ctr_aggregate.py

Requires apache-flink and the Kafka SQL connector jar. Does not install on
Windows/ARM64, which is why scripts/offline_ctr.py exists: it implements the
identical aggregation in plain Python so the skew number is computable on any
host, and CI runs both and asserts they agree.
"""
from __future__ import annotations

import argparse
import os
import struct

from pyflink.common import Types, WatermarkStrategy, Duration
from pyflink.common.serialization import SimpleStringSchema  # noqa: F401  (connector dep)
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaOffsetsInitializer,
    KafkaRecordDeserializationSchema,
)
from pyflink.datastream.formats.csv import CsvBulkWriters  # noqa: F401
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time

EVENT_BYTES = 17
EVENT_FMT = "<QIIB"


class RawDeserializer(KafkaRecordDeserializationSchema):
    """Passes the raw value bytes through; decoding happens in the map."""

    def deserialize(self, record, collector):  # pragma: no cover - runs in the TM
        if record.value and len(record.value) == EVENT_BYTES:
            collector.collect(record.value)

    def get_produced_type(self):
        return Types.PRIMITIVE_ARRAY(Types.BYTE())


def decode(raw: bytes):
    event_ms, user_id, item_id, etype = struct.unpack(EVENT_FMT, bytes(raw))
    return (int(item_id), 1, 1 if etype == 1 else 0, int(event_ms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brokers", default=os.environ.get("RECSERVE_BROKERS", "redpanda:9092"))
    ap.add_argument("--topic", default="interactions")
    ap.add_argument("--group", default="recserve-flink-ctr")
    ap.add_argument("--window-ms", type=int, default=60_000)
    ap.add_argument("--out", default="/opt/recserve/data/offline/flink_ctr.csv")
    ap.add_argument("--parallelism", type=int, default=2)
    args = ap.parse_args()

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(args.parallelism)
    env.get_config().set_auto_watermark_interval(200)

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(args.brokers)
        .set_topics(args.topic)
        .set_group_id(args.group)
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(RawDeserializer())
        .build()
    )

    # Event time comes from the record, not from arrival. 5 s of slack absorbs
    # producer clock skew and partition interleaving.
    watermarks = (
        WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5))
        .with_timestamp_assigner(lambda row, _: row[3])
    )

    decoded = (
        env.from_source(source, WatermarkStrategy.no_watermarks(), "interactions")
        .map(decode, output_type=Types.TUPLE([Types.INT(), Types.INT(), Types.INT(), Types.LONG()]))
        .assign_timestamps_and_watermarks(watermarks)
    )

    aggregated = (
        decoded.key_by(lambda r: r[0])
        .window(TumblingEventTimeWindows.of(Time.milliseconds(args.window_ms)))
        .reduce(lambda a, b: (a[0], a[1] + b[1], a[2] + b[2], max(a[3], b[3])))
        .map(
            lambda r: f"{r[0]},{r[1]},{r[2]},{(r[2] / r[1]) if r[1] else 0.0:.9f}",
            output_type=Types.STRING(),
        )
    )

    aggregated.sink_to(
        __import__("pyflink.datastream.connectors.file_system", fromlist=["FileSink"])
        .FileSink.for_row_format(args.out, SimpleStringSchema())
        .build()
    )

    env.execute("recserve-ctr-aggregate")


if __name__ == "__main__":
    main()
