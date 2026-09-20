#!/usr/bin/env python3
"""Bounded fixed-rate TCP load; latency starts at intended arrival, including client queue."""
import argparse
import asyncio
import collections
import datetime
import json
import math
import pathlib
import struct
import time


async def campaign(args):
    queue = asyncio.Queue(maxsize=args.queue)
    counts = collections.Counter()
    latency, scheduler_lag = [], []
    start = time.monotonic()
    attempted = math.ceil(args.rate * args.seconds)

    async def worker():
        while True:
            job = await queue.get()
            if job is None:
                queue.task_done()
                return
            number, intended = job
            writer = None
            try:
                remaining = args.deadline_ms / 1000 - (time.monotonic() - intended)
                if remaining <= 0:
                    counts['client_deadline'] += 1
                    continue
                async with asyncio.timeout(remaining):
                    reader, writer = await asyncio.open_connection(args.host, args.port)
                    writer.write(struct.pack('<IIQIIII', 0x52535631, 24, number, number % args.users,
                                             args.k, args.retrieve_k, max(1, int(remaining * 1e6))))
                    await writer.drain()
                    magic, length = struct.unpack('<II', await reader.readexactly(8))
                    if magic != 0x52535631 or not 16 <= length <= 4112:
                        raise ValueError('invalid frame')
                    response = await reader.readexactly(length)
                    rid, status, count = struct.unpack_from('<QII', response)
                    if rid != number or status > 4 or count > 512 or length != 16 + count * 8:
                        raise ValueError('invalid response')
                    counts[{0: 'ok', 1: 'server_timeout', 2: 'server_loadshed', 3: 'bad_request', 4: 'unavailable'}[status]] += 1
                    if status == 0:
                        latency.append((time.monotonic() - intended) * 1e6)
            except TimeoutError:
                counts['client_deadline'] += 1
            except (OSError, EOFError, asyncio.IncompleteReadError, ValueError):
                counts['transport_error'] += 1
            finally:
                if writer:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except OSError:
                        pass
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    for number in range(attempted):
        intended = start + number / args.rate
        await asyncio.sleep(max(0, intended - time.monotonic()))
        scheduler_lag.append((time.monotonic() - intended) * 1e6)
        try:
            queue.put_nowait((number, intended))
        except asyncio.QueueFull:
            counts['client_queue_drop'] += 1
    await queue.join()
    for _ in tasks:
        await queue.put(None)
    await asyncio.gather(*tasks)
    elapsed = time.monotonic()-start
    assert sum(counts.values()) == attempted
    def quantile(values, q):
        return sorted(values)[min(len(values)-1, math.ceil(q * len(values))-1)] if values else None
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), config=vars(args),
                protocol='fixed intended arrivals; one TCP connection per request; bounded client workers and queue',
                attempted=attempted, counts=dict(counts), elapsed_s=elapsed, achieved_success_qps=counts['ok']/elapsed,
                failure_rate=1-counts['ok']/attempted, success_p50_us=quantile(latency, .5),
                success_p99_us=quantile(latency, .99), scheduler_lag_p99_us=quantile(scheduler_lag, .99),
                latency_samples=len(latency))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9400)
    parser.add_argument('--rate', type=float, default=500)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--queue', type=int, default=256)
    parser.add_argument('--deadline-ms', type=float, default=100)
    parser.add_argument('--users', type=int, default=4096)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--retrieve-k', type=int, default=64)
    parser.add_argument('--out', default='.cache/logs/open-loop.json')
    args = parser.parse_args()
    if not (0 < args.rate <= 1000000 and 0 < args.seconds <= 3600 and 0 < args.deadline_ms <= 10000 and
            0 < args.concurrency <= 4096 and args.queue > 0 and args.users > 0 and 0 < args.k <= 512 and
            args.k <= args.retrieve_k <= 65536 and args.rate * args.seconds <= 10000000):
        parser.error('invalid load limits')
    result = asyncio.run(campaign(args))
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
