#!/usr/bin/env python3
"""Capacity and cost model: measured QPS and bytes -> hosts and dollars.

RSS and QPS on their own do not answer the question anyone actually asks, which
is "what does serving this catalog at this rate cost, and what changes if I
quantize". This turns the measured numbers into that answer.

The model, in full, so every number can be checked:

  cores_needed   = target_qps / (qps_per_core * utilisation)
  compute_hosts  = ceil(cores_needed / vcpus_per_host)
  bytes_per_item = embedding bytes + graph bytes (per dtype)
  memory_hosts   = ceil(catalog_bytes * replicas / usable_bytes_per_host)
  hosts          = max(compute_hosts, memory_hosts)
  $/hour         = hosts * price_per_hour
  $/M requests   = $/hour / (target_qps * 3600 / 1e6)

Two things this makes explicit that a QPS number hides:

  Serving is bound by whichever of compute and memory runs out first. A
  quantization that halves memory buys nothing if the fleet was already
  compute-bound -- the model reports which constraint binds.

  Utilisation is not 1.0. A host run at its measured saturation point has no
  headroom for a traffic spike, a deploy, or a failed peer, so capacity is
  sized at --utilisation (default 0.65).

Prices live in cost/instances.json with their source and date, because they go
stale and a hardcoded number in a script quietly becomes a lie.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exe(name: str) -> Path:
    for c in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
              ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    raise SystemExit(f"{name} not built")


def bench(kernel: str, args) -> dict:
    cmd = [str(exe("recserve_bench")), "--json", "--kernel", kernel,
           "--ef", str(args.ef), "--retrieve-k", str(args.retrieve_k), "--k", str(args.k),
           "--n", str(args.n), "--trials", str(args.trials), "--warmup", "64",
           "--recall-probe", str(args.recall_probe)]
    if args.catalog:
        cmd += ["--load-catalog", args.catalog]
        if args.index:
            cmd += ["--load-index", args.index]
    else:
        cmd += ["--items", str(args.items), "--dim", str(args.dim),
                "--clusters", str(args.clusters)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"bench failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def bytes_per_item(kernel: str, dim: int, m: int) -> dict[str, int]:
    """Embedding + graph bytes for one item, by kernel."""
    if kernel == "int8":
        emb = dim * 1 + 4          # int8 vector plus its float32 dequant scale
    elif kernel == "blocked":
        emb = dim * 4              # AoSoA is the same bytes, different order
    else:
        emb = dim * 4
    graph = min(m * 2, 128) * 4 + 4  # layer-0 adjacency plus the degree word
    return {"embedding": emb, "graph": graph, "total": emb + graph}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=["scalar", "simd", "int8"])
    ap.add_argument("--target-qps", type=float, default=1_000_000)
    ap.add_argument("--catalog-items", type=int, default=100_000_000,
                    help="items in the production catalog being sized for")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--utilisation", type=float, default=0.65)
    ap.add_argument("--slo-p99-us", type=float, default=0.0,
                    help="0 = derive from the scalar kernel's measured p99")
    ap.add_argument("--instances", default="cost/instances.json")
    # measurement workload
    ap.add_argument("--items", type=int, default=65536)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=1024)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--retrieve-k", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--recall-probe", type=int, default=128)
    ap.add_argument("--catalog", default="")
    ap.add_argument("--index", default="")
    ap.add_argument("--out", default="results/cost_model.json")
    args = ap.parse_args()

    inst = json.loads((ROOT / args.instances).read_text())
    rows = []
    measured = {kern: bench(kern, args) for kern in args.kernels}

    slo = args.slo_p99_us or measured[args.kernels[0]]["p99_mean_us"]
    print(f"SLO p99 = {slo:.1f} us (from {args.kernels[0]})   "
          f"target {args.target_qps:,.0f} qps   catalog {args.catalog_items:,} items   "
          f"replicas={args.replicas}  utilisation={args.utilisation}")
    print(f"prices: {inst['source']} ({inst['as_of']})\n")

    for kern, snap in measured.items():
        bpi = bytes_per_item(kern, args.dim, args.m)
        catalog_bytes = bpi["total"] * args.catalog_items
        qps_per_core = snap["qps_mean"]  # bench is single-threaded: 1 core
        meets_slo = snap["p99_mean_us"] <= slo

        for name, spec in inst["instances"].items():
            cores_needed = args.target_qps / (qps_per_core * args.utilisation)
            compute_hosts = math.ceil(cores_needed / spec["vcpu"])
            usable = spec["mem_gib"] * (1024 ** 3) * inst["usable_memory_frac"]
            memory_hosts = math.ceil(catalog_bytes * args.replicas / usable)
            hosts = max(compute_hosts, memory_hosts)
            bound = "memory" if memory_hosts > compute_hosts else "compute"
            usd_hr = hosts * spec["usd_per_hour"]
            per_million = usd_hr / (args.target_qps * 3600 / 1e6)
            rows.append({
                "kernel": kern, "instance": name,
                "p99_us": snap["p99_mean_us"], "meets_slo": meets_slo,
                "recall": snap["response_recall"],
                "qps_per_core": qps_per_core,
                "bytes_per_item": bpi["total"],
                "catalog_gib": catalog_bytes / (1024 ** 3),
                "compute_hosts": compute_hosts, "memory_hosts": memory_hosts,
                "hosts": hosts, "bound_by": bound,
                "usd_per_hour": usd_hr, "usd_per_month": usd_hr * 730,
                "usd_per_million_requests": per_million,
            })

    hdr = (f"{'kernel':8s} {'instance':14s} {'p99_us':>8s} {'slo':>4s} {'recall':>7s} "
           f"{'qps/core':>9s} {'B/item':>7s} {'cat_GiB':>9s} {'cpu_h':>6s} {'mem_h':>6s} "
           f"{'hosts':>6s} {'bound':>7s} {'$/hr':>9s} {'$/M req':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['kernel']:8s} {r['instance']:14s} {r['p99_us']:8.1f} "
              f"{'y' if r['meets_slo'] else 'n':>4s} {r['recall']:7.4f} "
              f"{r['qps_per_core']:9.0f} {r['bytes_per_item']:7d} {r['catalog_gib']:9.1f} "
              f"{r['compute_hosts']:6d} {r['memory_hosts']:6d} {r['hosts']:6d} "
              f"{r['bound_by']:>7s} {r['usd_per_hour']:9.2f} "
              f"{r['usd_per_million_requests']:8.4f}")

    # What quantization is actually worth, on the cheapest instance that meets SLO.
    savings = []
    for name in inst["instances"]:
        base = next((r for r in rows if r["kernel"] == "simd" and r["instance"] == name), None)
        q = next((r for r in rows if r["kernel"] == "int8" and r["instance"] == name), None)
        if base and q:
            savings.append({
                "instance": name,
                "hosts_f32": base["hosts"], "hosts_int8": q["hosts"],
                "usd_month_f32": base["usd_per_month"], "usd_month_int8": q["usd_per_month"],
                "usd_month_saved": base["usd_per_month"] - q["usd_per_month"],
                "recall_cost": base["recall"] - q["recall"],
                "bound_f32": base["bound_by"], "bound_int8": q["bound_by"],
            })
    print(f"\nint8 vs float32 at {args.target_qps:,.0f} qps, {args.catalog_items:,} items:")
    for s in savings:
        print(f"  {s['instance']:14s} {s['hosts_f32']:5d} -> {s['hosts_int8']:5d} hosts  "
              f"${s['usd_month_f32']:12,.0f} -> ${s['usd_month_int8']:12,.0f} /mo  "
              f"saves ${s['usd_month_saved']:11,.0f}/mo  "
              f"recall cost {s['recall_cost']:+.4f}  "
              f"({s['bound_f32']}-bound -> {s['bound_int8']}-bound)")

    payload = {"config": vars(args), "prices": inst, "measured": measured,
               "rows": rows, "int8_vs_f32": savings}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
