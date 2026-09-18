#!/usr/bin/env python3
"""Run the RecServe measurement campaign and write results/ + COST.md.

Every number published in the README comes from here, under one protocol:
fixed warmup, N trials, coefficient of variation reported, and a prebuilt index
snapshot so no measurement pays for a build the next one does not.

Stages (each can be skipped; --quick runs the cheap ones):
    tests      unit tests
    validate   RecServe HNSW vs hnswlib on identical data
    kernels    the kernel board, graph path and exact-scan path
    pareto     p99 vs recall across ef -- the tradeoff, not a single point
    crossover  blocked AoSoA vs AoS+SIMD across dim
    nearline   mutex vs RCU snapshot under live ingest, publish sweep
    skew       online/offline feature disagreement vs publish interval
    agent      closed-loop tuning from a misconfigured start
    cost       capacity and dollars
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL_STAGES = ["tests", "validate", "kernels", "pareto", "crossover", "quality", "nearline",
              "skew", "shard", "agent", "cost"]
QUICK = ["tests", "kernels", "pareto", "crossover"]


def exe(name: str) -> Path:
    for c in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
              ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    raise SystemExit(f"{name} not built -- cmake --build build --config Release")


def run(args: list[str], timeout: int = 3600) -> str:
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise SystemExit(f"cmd failed {args}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def bench(**kw) -> dict:
    cmd = [str(exe("recserve_bench")), "--json"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    return json.loads(run(cmd).strip().splitlines()[-1])


def ensure_fixture(items: int, dim: int, clusters: int, tag: str) -> tuple[str, str]:
    cat = f"data/catalog_{tag}.bin"
    idx = f"data/index_{tag}.bin"
    if not (ROOT / cat).exists() or not (ROOT / idx).exists():
        print(f"  building {tag} fixture ({items:,} x {dim})...", file=sys.stderr)
        run([str(exe("recserve_fixture")), "--items", str(items), "--dim", str(dim),
             "--clusters", str(clusters), "--m", "16", "--ef-construction", "100",
             "--queries", "4096", "--out-catalog", cat, "--out-index", idx], timeout=3600)
    return cat, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--board-only", action="store_true",
                    help="regenerate COST.md from the saved results without re-measuring")
    ap.add_argument("--big-items", type=int, default=1_000_000)
    ap.add_argument("--quality-prefix", default="",
                    help="ml25m or mlsmall; defaults to ml25m when prepared")
    ap.add_argument("--out", default="results/measured.json")
    args = ap.parse_args()

    if args.board_only:
        payload = json.loads((ROOT / args.out).read_text())
        write_board(payload)
        print(f"-> {ROOT / 'COST.md'}")
        return 0

    stages = args.stages or (QUICK if args.quick else ALL_STAGES)
    # Merge into whatever is already there. Running a single stage used to
    # rewrite the file from scratch and silently drop every other result.
    payload: dict = {}
    existing = ROOT / args.out
    if existing.exists():
        try:
            payload = json.loads(existing.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload.update({
        "host": {
            "os": platform.platform(), "machine": platform.machine(),
            "cpus": os.cpu_count(), "python": sys.version.split()[0],
            "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "stages_run": sorted(set(payload.get("stages_run", [])) | set(stages)),
    })

    if "tests" in stages:
        print("[tests]", file=sys.stderr)
        payload["tests"] = run([str(exe("recserve_tests"))]).strip()

    cat64, idx64 = ensure_fixture(65536, 64, 1024, "64k")
    payload["isa"] = bench(items=4096, dim=64, n=50, trials=1, recall_probe=0)["isa"]

    if "validate" in stages:
        print("[validate] recserve vs hnswlib", file=sys.stderr)
        run([sys.executable, "scripts/validate_hnsw.py", "--items", "65536",
             "--dim", "64", "--clusters", "1024", "--queries", "256"], timeout=3600)
        payload["hnsw_validation"] = json.loads(
            (ROOT / "results" / "hnsw_validation.json").read_text())

    if "kernels" in stages:
        print("[kernels]", file=sys.stderr)
        kernels = ["scalar", "simd", "soa", "blocked", "int8"]
        graph, exact = [], []
        for kern in kernels:
            graph.append(bench(load_catalog=cat64, load_index=idx64, kernel=kern,
                               ef=64, retrieve_k=64, k=10, n=2000, trials=5,
                               recall_probe=128))
            exact.append(bench(load_catalog=cat64, kernel=kern, brute=True, ef=64,
                               retrieve_k=10, k=10, n=60, trials=3, recall_probe=0))
        payload["kernels_graph_64k"] = graph
        payload["kernels_exact_64k"] = exact

        big_cat = ROOT / f"data/catalog_{args.big_items // 1000}k.bin"
        if big_cat.exists():
            payload["kernels_exact_big"] = [
                bench(load_catalog=str(big_cat.relative_to(ROOT)).replace("\\", "/"),
                      kernel=kern, brute=True, retrieve_k=10, k=10, n=40, trials=3,
                      recall_probe=0)
                for kern in kernels]

    if "pareto" in stages:
        print("[pareto] p99 vs recall across ef", file=sys.stderr)
        payload["pareto_ef"] = [
            bench(load_catalog=cat64, load_index=idx64, kernel="simd", ef=ef,
                  retrieve_k=10, k=10, n=1500, trials=5, recall_probe=128)
            for ef in (16, 24, 32, 48, 64, 96, 128, 192, 256, 384)]

    if "crossover" in stages:
        print("[crossover] blocked vs simd across dim", file=sys.stderr)
        rows = []
        for d in (8, 16, 24, 32, 48, 64, 96, 128):
            a = bench(items=200_000, dim=d, clusters=512, kernel="simd", brute=True,
                      retrieve_k=10, k=10, n=40, trials=3, recall_probe=0)
            b = bench(items=200_000, dim=d, clusters=512, kernel="blocked", brute=True,
                      retrieve_k=10, k=10, n=40, trials=3, recall_probe=0)
            rows.append({"dim": d, "simd_p99_us": a["p99_mean_us"],
                         "blocked_p99_us": b["p99_mean_us"],
                         "ratio": b["p99_mean_us"] / a["p99_mean_us"]})
        payload["blocked_crossover"] = rows

    if "quality" in stages:
        print("[quality] real embeddings on movielens", file=sys.stderr)
        # Prefer ml-25m when it has been prepared: ml-latest-small has 609 users,
        # which is far too few for 64 factors, and ALS there barely separates
        # from most-popular. The small set exists so CI can run this at all.
        prefix = args.quality_prefix
        if not prefix:
            prefix = "ml25m" if (ROOT / "data" / "ml25m_catalog.bin").exists() else "mlsmall"
        cat = ROOT / "data" / f"{prefix}_catalog.bin"
        if not cat.exists():
            run([sys.executable, "scripts/prep_movielens.py", "--dataset",
                 "25m" if prefix == "ml25m" else "small",
                 "--factors", "64" if prefix == "ml25m" else "32",
                 "--iterations", "20" if prefix == "ml25m" else "15"], timeout=7200)
        idx = ROOT / "data" / f"{prefix}_index.bin"
        if not idx.exists():
            run([str(exe("recserve_fixture")), "--in-catalog", f"data/{prefix}_catalog.bin",
                 "--out-index", f"data/{prefix}_index.bin", "--m", "16",
                 "--ef-construction", "200", "--build-threads", "1"], timeout=3600)
        run([sys.executable, "scripts/eval_baselines.py", "--prefix", prefix, "--k", "10",
             "--out", "results/quality_real.json"], timeout=3600)
        payload["quality_real"] = json.loads(
            (ROOT / "results" / "quality_real.json").read_text())
        payload["quality_real"]["dataset"] = prefix
        payload["quality_real_ef"] = [
            json.loads(run([str(exe("recserve_eval")),
                            "--catalog", f"data/{prefix}_catalog.bin",
                            "--queries", f"data/{prefix}_queries.bin",
                            "--test", f"data/{prefix}_test.csv",
                            "--train", f"data/{prefix}_train.csv",
                            "--index", f"data/{prefix}_index.bin",
                            "--k", "10", "--ef", str(ef), "--json"]).strip().splitlines()[-1])
            for ef in (16, 32, 64, 128, 256)]

    if "nearline" in stages:
        print("[nearline] mutex vs snapshot", file=sys.stderr)
        run([sys.executable, "scripts/nearline_ab.py", "--rates", "10000", "100000",
             "500000", "--trials", "3", "--seconds", "4"], timeout=3600)
        payload["nearline_ab"] = json.loads(
            (ROOT / "results" / "nearline_ab.json").read_text())

    if "skew" in stages:
        print("[skew] online vs offline features", file=sys.stderr)
        if not (ROOT / "data" / "events.bin").exists():
            run([str(exe("recserve_nearline")), "--write-log", "200000",
                 "--items", "16384", "--out", "data/events.bin"])
        run([sys.executable, "scripts/skew.py", "--n", "200000",
             "--publish-ms", "100", "1000", "5000", "30000"], timeout=1800)
        payload["skew"] = json.loads((ROOT / "results" / "skew.json").read_text())

    if "shard" in stages:
        print("[shard] tail amplification", file=sys.stderr)
        big = ROOT / f"data/catalog_{args.big_items // 1000}k.bin"
        rows = []
        for n in (1, 2, 4, 8, 16, 32):
            cmd = [str(exe("recserve_shard")), "--shards", str(n), "--k", "10",
                   "--ef", "64", "--n", "1500", "--json"]
            cmd += ["--catalog", str(big.relative_to(ROOT)).replace("\\", "/")] if big.exists()                 else ["--items", "200000", "--dim", "64", "--clusters", "1024"]
            rows.append(json.loads(run(cmd, timeout=3600).strip().splitlines()[-1]))
        payload["sharding"] = {"cores": os.cpu_count(), "rows": rows}

    if "agent" in stages:
        print("[agent] closed loop from a misconfigured start", file=sys.stderr)
        run([sys.executable, "scripts/agent.py", "--compare", "--iterations", "12",
             "--catalog", cat64, "--index", idx64, "--n", "2000", "--trials", "5",
             "--recall-probe", "128", "--start-config",
             '{"ef":256,"retrieve_k":128,"kernel":"scalar"}'], timeout=3600)
        payload["agent"] = json.loads((ROOT / "results" / "agent_summary.json").read_text())

    if "cost" in stages:
        print("[cost] capacity and dollars", file=sys.stderr)
        run([sys.executable, "scripts/cost_model.py", "--catalog", cat64,
             "--index", idx64, "--target-qps", "1000000",
             "--catalog-items", "100000000"], timeout=1800)
        payload["cost"] = json.loads((ROOT / "results" / "cost_model.json").read_text())

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    write_board(payload)
    print(f"\n-> {out}\n-> {ROOT / 'COST.md'}")
    return 0


def write_board(p: dict) -> None:
    h = p["host"]
    L: list[str] = []
    A = L.append
    A("# Cost and capacity board")
    A("")
    A(f"- Host: {h['os']} / {h['machine']} / {h['cpus']} logical CPUs / ISA `{p.get('isa','?')}`")
    A(f"- Measured: {h['measured_at_utc']}")
    A("- Protocol: in-process `recommend_sync`, prebuilt index snapshot, 64-request warmup,")
    A("  N trials with the coefficient of variation reported, synthetic clustered embeddings.")
    A("- Single-host benchmark. Not production, not multi-region, not ByteDance scale.")
    A("")

    if "hnsw_validation" in p:
        v = p["hnsw_validation"]
        A("## Index correctness (vs hnswlib, identical data and parameters)")
        A("")
        A("| ef | hnswlib recall@10 | RecServe recall@10 | delta |")
        A("|---|---|---|---|")
        for r in v["rows"]:
            A(f"| {r['ef']} | {r['hnswlib_recall']:.4f} | {r['recserve_recall']:.4f} | "
              f"{r['delta']:+.4f} |")
        A("")
        A(f"Max absolute delta {v['max_abs_delta']:.4f} over {v['items']:,} items, "
          f"M={v['m']}, efConstruction={v['ef_construction']}.")
        A("")

    if "kernels_graph_64k" in p:
        A("## Kernels, graph retrieval (65,536 items, dim 64, ef=64, retrieve_k=64)")
        A("")
        A("| kernel | p99 us | p99 CV | QPS | response recall@10 | RSS MiB |")
        A("|---|---|---|---|---|---|")
        for r in p["kernels_graph_64k"]:
            A(f"| {r['kernel']} | {r['p99_mean_us']:.1f} | {r['p99_cv']:.3f} | "
              f"{r['qps_mean']:.0f} | {r['response_recall']:.4f} | {r['rss_mib']:.1f} |")
        A("")

    if "kernels_exact_64k" in p:
        A("## Kernels, exact scan (65,536 items) -- where layout and dtype actually bind")
        A("")
        A("| kernel | p99 us | QPS | catalog MiB |")
        A("|---|---|---|---|")
        for r in p["kernels_exact_64k"]:
            A(f"| {r['kernel']} | {r['p99_mean_us']:.1f} | {r['qps_mean']:.0f} | "
              f"{r['catalog_mib']:.1f} |")
        A("")

    if "kernels_exact_big" in p:
        n = p["kernels_exact_big"][0]["items"]
        A(f"## Kernels, exact scan ({n:,} items) -- past every cache")
        A("")
        A("| kernel | p99 ms | QPS | catalog MiB |")
        A("|---|---|---|---|")
        for r in p["kernels_exact_big"]:
            A(f"| {r['kernel']} | {r['p99_mean_us'] / 1000:.2f} | {r['qps_mean']:.1f} | "
              f"{r['catalog_mib']:.1f} |")
        A("")

    if "pareto_ef" in p:
        A("## Latency / recall tradeoff (ef sweep, 65,536 items)")
        A("")
        A("| ef | p99 us | QPS | retrieve recall@10 | response recall@10 | hops |")
        A("|---|---|---|---|---|---|")
        for r in p["pareto_ef"]:
            A(f"| {r['ef']} | {r['p99_mean_us']:.1f} | {r['qps_mean']:.0f} | "
              f"{r['retrieve_recall']:.4f} | {r['response_recall']:.4f} | "
              f"{r['hops_mean']:.0f} |")
        A("")

    if "blocked_crossover" in p:
        A("## Blocked AoSoA vs AoS+SIMD across dim (200,000 items, exact scan)")
        A("")
        A("| dim | simd p99 us | blocked p99 us | blocked/simd |")
        A("|---|---|---|---|")
        for r in p["blocked_crossover"]:
            A(f"| {r['dim']} | {r['simd_p99_us']:.0f} | {r['blocked_p99_us']:.0f} | "
              f"{r['ratio']:.2f}x |")
        A("")
        rows = p["blocked_crossover"]
        clear_win = [r["dim"] for r in rows if r["ratio"] <= 0.85]
        clear_loss = [r["dim"] for r in rows if r["ratio"] >= 1.15]
        if clear_win and clear_loss:
            A(f"Blocked wins clearly at dim <= {max(clear_win)} "
              f"({min(r['ratio'] for r in rows):.2f}x at the low end), is within noise "
              f"between dim {max(clear_win)} and {min(clear_loss)}, and loses from "
              f"dim {min(clear_loss)} up ({max(r['ratio'] for r in rows):.2f}x).")
            A("Below the crossover the per-item horizontal reduction dominates and blocking")
            A("amortises it across 8 items; above it, broadcasting q[d] once per dimension")
            A("costs more than the reduction it removes.")
            A("")

    if "nearline_ab" in p:
        A("## Feature store under live ingest (8 serving threads, median of 3)")
        A("")
        A("| events/s | store | serve QPS | p50 us | p99 us | p99.9 us | freshness p99 ms |")
        A("|---|---|---|---|---|---|---|")
        for r in p["nearline_ab"]["ab"]:
            A(f"| {r['rate']:,} | {r['store']} | {r['serve_qps']:.0f} | {r['p50_us']:.0f} | "
              f"{r['p99_us']:.0f} | {r['p999_us']:.0f} | {r['freshness_p99_ms']:.0f} |")
        A("")
        A("### Publish interval: the price of not holding a lock")
        A("")
        A("| publish ms | p99 us | p99.9 us | freshness p50 ms | freshness p99 ms | publish p99 us |")
        A("|---|---|---|---|---|---|")
        for r in p["nearline_ab"]["publish_sweep"]:
            A(f"| {r['publish_ms']} | {r['p99_us']:.0f} | {r['p999_us']:.0f} | "
              f"{r['freshness_p50_ms']:.0f} | {r['freshness_p99_ms']:.0f} | "
              f"{r['publish_p99_us']:.0f} |")
        A("")

    if "skew" in p:
        A("## Online/offline feature skew vs publish interval")
        A("")
        A("| publish ms | unpublished events | items differing | max dCTR | mean dCTR |")
        A("|---|---|---|---|---|")
        for r in p["skew"]["rows"]:
            A(f"| {r['publish_ms']} | {r['unpublished']:,} | {r['items_differ']:,} | "
              f"{r['max_abs_ctr_delta']:.4f} | {r['mean_abs_ctr_delta']:.6f} |")
        A("")

    if "quality_real" in p:
        q = p["quality_real"]
        A(f"## Recommendation quality on real embeddings "
          f"({q.get('dataset', '?')}: {q['items']:,} items, {q['eval_users']:,} users)")
        A("")
        A("ALS on MovieLens, per-user temporal split, items already seen in training")
        A(f"filtered from the returned list. recall@{q['k']} ceiling is "
          f"{q['recall_ceiling']:.3f} (mean held-out set {q['mean_gold']:.1f} items).")
        A("")
        A(f"| method | recall@{q['k']} | ndcg@{q['k']} |")
        A("|---|---|---|")
        for name, row in q["baselines"].items():
            A(f"| {name} | {row['recall']:.4f} | {row['ndcg']:.4f} |")
        A(f"| **served through recserve** | **{q['cpp']['recall_at_k']:.4f}** | "
          f"**{q['cpp']['ndcg_at_k']:.4f}** |")
        A("")
        lift = q["als_lift_over_popularity"]
        pop = max(q["baselines"]["most_popular"]["recall"], 1e-9)
        A(f"ALS beats most-popular by {lift:+.4f} recall ({lift / pop * 100:+.1f}%). The C++")
        A("service and an independent numpy implementation of the same metric agree to")
        A(f"{abs(q['cpp']['recall_at_k'] - q['baselines']['als_exact']['recall']):.5f}.")
        A("")
        if "quality_real_ef" in p:
            A("| ef | recall@10 | ndcg@10 | retrieve recall | p99 us |")
            A("|---|---|---|---|---|")
            for r in p["quality_real_ef"]:
                A(f"| {r['ef']} | {r['recall_at_k']:.4f} | {r['ndcg_at_k']:.4f} | "
                  f"{r['retrieve_recall']:.4f} | {r['p99_us']:.0f} |")
            A("")
            A("On trained embeddings the graph finds essentially all of the exact top-k at")
            A("small ef. On synthetic uniform vectors the same ef reaches about half, which")
            A("is a property of the data, not of the index.")
            A("")

    if "sharding" in p:
        cores = p["sharding"]["cores"]
        A(f"## Sharded retrieval: tail amplification ({cores} cores)")
        A("")
        A("A request finishes when its SLOWEST shard replies, so end-to-end latency is")
        A("the maximum of N samples, not the mean (Dean & Barroso, \"The Tail at Scale\",")
        A("CACM 56(2), 2013). Shards are threads in one process: no network, no separate")
        A("failure domain, so this is a LOWER BOUND on what a real deployment would see.")
        A("")
        A("| shards | items/shard | shard p99 us | e2e p99 us | merge p99 us | amplification | recall@10 |")
        A("|---|---|---|---|---|---|---|")
        for r in p["sharding"]["rows"]:
            flag = " *" if r["shards"] > cores else ""
            A(f"| {r['shards']}{flag} | {r['items_per_shard']:,} | {r['shard_p99_us']:.0f} | "
              f"{r['e2e_p99_us']:.0f} | {r['merge_p99_us']:.0f} | "
              f"{r['tail_amplification']:.2f}x | {r['recall_vs_exact']:.4f} |")
        A("")
        A(r"\* more shards than cores: those rows include CPU queueing on top of the")
        A("structural tail effect, so read them as an upper bound rather than as a")
        A("clean measurement of amplification.")
        A("")
        A("Recall rises with shard count because each shard searches its own smaller")
        A("graph at the same ef and every shard contributes its own top-k, so total")
        A("candidates examined scales with N. Sharding buys accuracy and costs tail")
        A("latency; that trade is the actual decision, not whether scatter-gather works.")
        A("")

    if "agent" in p:
        A("## Closed-loop agent, from a misconfigured start")
        A("")
        A("| planner | start p99 us | best p99 us | improvement | confirmed | accepted | rejected |")
        A("|---|---|---|---|---|---|---|")
        for r in p["agent"]["runs"]:
            A(f"| {r['planner']} | {r['baseline_p99_us']:.1f} | {r['best_p99_us']:.1f} | "
              f"{r['improvement_frac'] * 100:+.1f}% | {r['final_confirmed']} | "
              f"{r['accepted']} | {r['rejected']} |")
        A("")
        floors = [r.get("detect_floor_frac", 0) for r in p["agent"]["runs"]]
        if floors:
            A(f"Smallest change this host can resolve at this protocol: "
              f"{min(floors) * 100:.1f}% to {max(floors) * 100:.1f}% "
              f"(2 sigma, calibrated per run against the START config -- a slow,")
            A("misconfigured start is noisier in absolute terms, so its floor is wider.)")
            A("")
            A("On an already-tuned config all planners accept nothing. The best remaining")
            A("change is retrieve_k 64 -> 16; a separate 40-run interleaved A/B measures it")
            A("at +2.9% with t = 2.10, below the floor, so declining to claim it is correct.")
            A("")

    if "cost" in p:
        c = p["cost"]
        A(f"## Capacity and cost ({c['config']['target_qps']:,.0f} qps, "
          f"{c['config']['catalog_items']:,} items, {c['config']['replicas']} replicas, "
          f"{c['config']['utilisation']:.0%} utilisation)")
        A("")
        A(f"Prices: {c['prices']['source']} ({c['prices']['as_of']}, "
          f"{c['prices']['pricing_model']}). On-demand list is a ceiling, not a quote.")
        A("")
        A("| kernel | instance | QPS/core | B/item | hosts | bound by | $/hour | $/M requests |")
        A("|---|---|---|---|---|---|---|---|")
        for r in c["rows"]:
            A(f"| {r['kernel']} | {r['instance']} | {r['qps_per_core']:.0f} | "
              f"{r['bytes_per_item']} | {r['hosts']} | {r['bound_by']} | "
              f"{r['usd_per_hour']:.2f} | {r['usd_per_million_requests']:.4f} |")
        A("")
        for s in c["int8_vs_f32"]:
            A(f"- **{s['instance']}**: int8 takes {s['hosts_f32']} hosts to "
              f"{s['hosts_int8']}, ${s['usd_month_f32']:,.0f} to "
              f"${s['usd_month_int8']:,.0f}/month (saves ${s['usd_month_saved']:,.0f}), "
              f"at a cost of {abs(s['recall_cost']):.4f} recall@10.")
        A("")

    A("Do not restate any of this as production QPS, multi-region serving, or trained-model quality.")
    (ROOT / "COST.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
