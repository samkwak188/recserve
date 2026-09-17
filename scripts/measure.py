#!/usr/bin/env python3
"""Run the RecServe measurement campaign and write results/measured.json + COST.md."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "Release"
if not (BIN / "recserve_bench.exe").exists() and not (BIN / "recserve_bench").exists():
    BIN = ROOT / "build"


def exe(name: str) -> Path:
    p = BIN / (name + (".exe" if os.name == "nt" else ""))
    if not p.exists():
        p = BIN / name
    return p


def run(args: list[str], timeout: int = 180) -> str:
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"cmd failed {args}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main() -> None:
    host = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    tests = run([str(exe("recserve_tests"))])
    q_f32 = json.loads(run([str(exe("recserve_quality")), "--csv", "data/quality_fixture.csv", "--k", "5", "--json"]))
    q_i8 = json.loads(
        run([str(exe("recserve_quality")), "--csv", "data/quality_fixture.csv", "--k", "5", "--json", "--int8"])
    )
    modes = ["baseline", "arena", "soa", "simd", "int8", "pin"]
    scale = []
    for items in (4096, 16384):
        for mode in modes:
            out = run(
                [
                    str(exe("recserve_bench")),
                    "--json",
                    "--items",
                    str(items),
                    "--dim",
                    "64",
                    "--n",
                    "600",
                    "--k",
                    "10",
                    "--retrieve-k",
                    "64",
                    "--mode",
                    mode,
                    "--trials",
                    "3",
                ],
                timeout=300,
            )
            row = json.loads(out.strip().splitlines()[-1])
            row["catalog"] = items
            scale.append(row)
    near = run([str(exe("recserve_nearline")), "--n", "5000", "--out", "data/events.bin", "--delay-us", "20000"])
    soak = run([str(exe("recserve_soak")), "--seconds", "3", "--items", "2048", "--dim", "32"])
    diag = run([str(exe("recserve_diagnose")), "--p99", "1000", "--queue-p99", "800", "--score-p99", "100"])

    payload = {
        "host": host,
        "tests": tests.strip(),
        "quality_f32": q_f32,
        "quality_int8": q_i8,
        "scale": scale,
        "nearline_stdout": near.strip(),
        "soak_stdout": soak.strip(),
        "diagnose_stdout": diag.strip(),
    }
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / "measured.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_cost(payload)
    print(json.dumps({"wrote": str(outdir / "measured.json"), "scale_rows": len(scale)}, indent=2))


def write_cost(p: dict) -> None:
    host = p["host"]
    rows_4k = [r for r in p["scale"] if r["catalog"] == 4096]
    rows_16k = [r for r in p["scale"] if r["catalog"] == 16384]
    simd = next(r for r in rows_16k if r["mode"] == "simd")
    lines = [
        "# Cost and capacity board",
        "",
        f"- Host: {host['os']} / {host['machine']} / {host['cpus']} logical CPUs",
        f"- Measured: {host['measured_at_utc']}",
        "- Protocol: in-process `recommend_sync`, 3 trials, 64-request warmup, catalog = synthetic unit-normalized embeddings",
        "- This is a single-host benchmark, not production serving.",
        "",
        "## Quality gate (data/quality_fixture.csv, k=5)",
        "",
        f"- float32: recall={p['quality_f32']['recall']:.4f}, NDCG={p['quality_f32']['ndcg']:.4f}, retrieve-recall={p['quality_f32']['retrieve_recall']:.4f}, users={p['quality_f32']['users']}",
        f"- int8: recall={p['quality_int8']['recall']:.4f}, NDCG={p['quality_int8']['ndcg']:.4f}, retrieve-recall={p['quality_int8']['retrieve_recall']:.4f}",
        "",
        "## Engine throughput (16384 items, dim=64, retrieve_k=64, k=10, n=600)",
        "",
        "| mode | p99 mean us | p99 CV | QPS | RSS MiB |",
        "|---|---|---|---|---|",
    ]
    for r in rows_16k:
        lines.append(
            f"| {r['mode']} | {r['p99_mean_us']:.2f} | {r['p99_cv']:.3f} | {r['qps_mean']:.0f} | {r['rss_mib']:.1f} |"
        )
    lines += [
        "",
        "## Same protocol, 4096-item catalog",
        "",
        "| mode | p99 mean us | p99 CV | QPS | RSS MiB |",
        "|---|---|---|---|---|",
    ]
    for r in rows_4k:
        lines.append(
            f"| {r['mode']} | {r['p99_mean_us']:.2f} | {r['p99_cv']:.3f} | {r['qps_mean']:.0f} | {r['rss_mib']:.1f} |"
        )
    base = next(r for r in rows_16k if r["mode"] == "baseline")
    slo_us = base["p99_mean_us"] * 1.10
    lines += [
        "",
        f"## Predeclared SLO (from baseline p99, not reverse-picked)",
        "",
        f"- Baseline p99 mean = {base['p99_mean_us']:.2f} us on the 16384-item catalog.",
        f"- SLO = 1.10x baseline = **{slo_us:.2f} us p99**. SIMD mode p99 mean = {simd['p99_mean_us']:.2f} us ({'PASS' if simd['p99_mean_us'] <= slo_us else 'FAIL'}).",
        "",
        "## Nearline / soak",
        "",
        f"- `{p['nearline_stdout']}`",
        f"- `{p['soak_stdout']}`",
        "",
        "Do not rewrite these as production QPS, multi-region serving, or CHTC results.",
        "",
    ]
    (ROOT / "COST.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
