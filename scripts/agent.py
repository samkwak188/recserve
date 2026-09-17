#!/usr/bin/env python3
"""Closed-loop tuning agent: observe, diagnose, plan, execute, verify, improve.

The loop is the point, not the model. Every proposal -- whether it came from
Claude, from a rule, or from a coin flip -- goes through the same verification
gate, and anything that fails is rolled back. That is what makes this an
optimizer rather than a chatbot with shell access.

  observe   run recserve_bench under the current config; one JSON snapshot with
            latency percentiles, the per-stage split, recall, RSS and hop count
  diagnose  rank hypotheses from the snapshot, each with the evidence for it
  plan      pick ONE knob and ONE direction from an enumerated action space
  execute   re-run the benchmark with the new config
  verify    accept only if p99 improves by more than the combined noise band
            AND recall stays above the floor AND RSS stays under budget
  improve   append (state, action, outcome) to results/agent_runs.jsonl and
            feed the history back so disproved actions are not re-proposed

The planner is a strategy, so the same loop runs with --planner llm, heuristic,
random or grid. --compare runs all four on the same budget from the same start,
which is the only honest way to claim the LLM planner is doing anything: a
tuner that cannot beat random search on the same budget has not earned its
inference cost.

Safety properties, in order of how much they matter:

  1. The LLM never writes code, never runs a command and never touches the
     serving path. It returns one (knob, value) pair, which is validated
     against the enumerated space before anything runs. An unparseable or
     out-of-space proposal is discarded and the heuristic planner takes the
     turn.
  2. Verification uses the same protocol as the published board -- same warmup,
     same trial count -- so an accepted change cannot be an artifact of
     comparing two different measurement protocols.
  3. Every accepted change must clear the noise band. An improvement smaller
     than the measurement error is not an improvement.
  4. Rollback is automatic and unconditional on failure.

Usage:
    python scripts/agent.py --iterations 12 --planner heuristic
    python scripts/agent.py --iterations 12 --planner llm       # ANTHROPIC_API_KEY
    python scripts/agent.py --planner llm --provider gemini     # GEMINI_API_KEY
    python scripts/agent.py --compare --iterations 10
    python scripts/agent.py --validate-pr                       # CI gate mode
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Action space. The agent may only ever move within this. Adding a knob here is
# a deliberate act; the planner cannot invent one.
# --------------------------------------------------------------------------
ACTION_SPACE: dict[str, list[Any]] = {
    "ef": [16, 24, 32, 48, 64, 96, 128, 192, 256],
    "retrieve_k": [10, 16, 24, 32, 48, 64, 96, 128],
    "kernel": ["scalar", "simd", "blocked", "int8"],
    "pin": [0, 1],
}

DEFAULT_CONFIG: dict[str, Any] = {"ef": 64, "retrieve_k": 64, "kernel": "simd", "pin": 0}

# The knobs are not independent: the engine searches with
# max(ef_search, retrieve_k), so any ef below the current retrieve_k is a no-op
# on both latency and recall. A sweep of ef at retrieve_k=64 therefore looks
# completely flat below 64 -- which is what the grid arm found, and why a
# verifier that trusted a single paired measurement accepted one of those
# no-ops as a 10% win.


def exe(name: str) -> Path:
    for c in (ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / name,
              ROOT / "build" / "Release" / name, ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    raise SystemExit(f"{name} not built -- cmake --build build --config Release")


# --------------------------------------------------------------------------
# 1. OBSERVE
# --------------------------------------------------------------------------
@dataclass
class Workload:
    items: int = 65536
    dim: int = 64
    clusters: int = 1024
    n: int = 2000
    k: int = 10
    trials: int = 5
    warmup: int = 64
    recall_probe: int = 96
    catalog: str = ""
    index: str = ""


def observe(config: dict[str, Any], w: Workload) -> dict[str, Any]:
    """One measured snapshot of the system under `config`."""
    cmd = [str(exe("recserve_bench")), "--json",
           "--n", str(w.n), "--k", str(w.k), "--trials", str(w.trials),
           "--warmup", str(w.warmup), "--recall-probe", str(w.recall_probe),
           "--kernel", str(config["kernel"]), "--ef", str(config["ef"]),
           "--retrieve-k", str(config["retrieve_k"])]
    if config.get("pin"):
        cmd.append("--pin")
    if w.catalog:
        cmd += ["--load-catalog", w.catalog]
        if w.index:
            cmd += ["--load-index", w.index]
    else:
        cmd += ["--items", str(w.items), "--dim", str(w.dim), "--clusters", str(w.clusters)]

    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"bench failed: {' '.join(cmd)}\n{out.stdout}\n{out.stderr}")
    snap = json.loads(out.stdout.strip().splitlines()[-1])
    snap["config"] = dict(config)
    # Uncertainty of the REPORTED p99, which is a mean over trials -- so the
    # standard error of that mean, s/sqrt(trials), not the raw trial-to-trial
    # standard deviation. Using the raw SD would treat the spread of individual
    # trials as the error of their average and reject real wins forever.
    sd = snap["p99_mean_us"] * snap["p99_cv"]
    snap["p99_sigma_us"] = sd / math.sqrt(max(1, w.trials))
    snap["p99_sd_us"] = sd
    return snap


# --------------------------------------------------------------------------
# 2. DIAGNOSE
# --------------------------------------------------------------------------
@dataclass
class Hypothesis:
    name: str
    confidence: float
    evidence: dict[str, Any]
    suggests: str  # which knob it implicates


def diagnose(snap: dict[str, Any], recall_floor: float) -> list[Hypothesis]:
    """Ranked hypotheses with the evidence for each. Not a single enum."""
    hs: list[Hypothesis] = []
    p99 = max(snap["p99_mean_us"], 1e-9)
    retr = snap["retrieve_p99_us"]
    score = snap["score_p99_us"]
    feat = snap["feature_p99_us"]
    recall = snap["retrieve_recall"]
    headroom = recall - recall_floor

    if retr > 0.5 * p99:
        hs.append(Hypothesis(
            "retrieve_dominates", min(1.0, retr / p99),
            {"retrieve_p99_us": retr, "share_of_p99": round(retr / p99, 3),
             "ef": snap["ef"], "hops_mean": snap["hops_mean"],
             "recall": recall, "recall_headroom": round(headroom, 4)},
            "ef" if headroom > 0.01 else "kernel"))

    if score > 0.35 * p99:
        hs.append(Hypothesis(
            "scoring_dominates", min(1.0, score / p99),
            {"score_p99_us": score, "share_of_p99": round(score / p99, 3),
             "retrieve_k": snap["retrieve_k"], "kernel": snap["kernel"]},
            "retrieve_k"))

    if feat > 0.25 * p99:
        hs.append(Hypothesis(
            "feature_store_wait", min(1.0, feat / p99),
            {"feature_p99_us": feat, "share_of_p99": round(feat / p99, 3)},
            "kernel"))

    if snap["p99_cv"] > 0.20:
        hs.append(Hypothesis(
            "measurement_unstable", min(1.0, snap["p99_cv"]),
            {"p99_cv": snap["p99_cv"],
             "note": "trial spread exceeds 20%; small wins here are not real"},
            "pin"))

    if headroom > 0.05:
        hs.append(Hypothesis(
            "recall_headroom", min(1.0, headroom * 4),
            {"recall": recall, "floor": recall_floor, "headroom": round(headroom, 4),
             "note": "accuracy above the floor can be traded for latency"},
            "ef"))
    elif headroom < 0.0:
        hs.append(Hypothesis(
            "recall_violation", 1.0,
            {"recall": recall, "floor": recall_floor}, "ef"))

    hs.sort(key=lambda h: h.confidence, reverse=True)
    return hs


# --------------------------------------------------------------------------
# 3. PLAN -- four interchangeable strategies over the same action space
# --------------------------------------------------------------------------
@dataclass
class Action:
    knob: str
    value: Any
    rationale: str
    source: str

    def valid(self) -> bool:
        return self.knob in ACTION_SPACE and self.value in ACTION_SPACE[self.knob]


def _neighbour(knob: str, cur: Any, direction: int) -> Any:
    space = ACTION_SPACE[knob]
    i = space.index(cur) if cur in space else 0
    return space[max(0, min(len(space) - 1, i + direction))]


def plan_heuristic(snap, hyps, history, cfg, rng) -> Action:
    """Rules derived from the diagnosis. Also the fallback for every planner."""
    tried = {(h["action"]["knob"], h["action"]["value"]) for h in history}
    for h in hyps:
        if h.name == "recall_violation":
            v = _neighbour("ef", cfg["ef"], +1)
            if (("ef", v)) not in tried and v != cfg["ef"]:
                return Action("ef", v, "recall below floor; widen the beam", "heuristic")
        if h.name in ("retrieve_dominates", "recall_headroom") and h.evidence.get(
                "recall_headroom", h.evidence.get("headroom", 0)) > 0.01:
            v = _neighbour("ef", cfg["ef"], -1)
            if (("ef", v)) not in tried and v != cfg["ef"]:
                return Action("ef", v, "retrieve dominates and recall has headroom; "
                                       "narrow the beam", "heuristic")
        if h.name == "scoring_dominates":
            v = _neighbour("retrieve_k", cfg["retrieve_k"], -1)
            if (("retrieve_k", v)) not in tried and v != cfg["retrieve_k"]:
                return Action("retrieve_k", v, "ranking cost scales with candidate count",
                              "heuristic")
        if h.name == "measurement_unstable" and not cfg["pin"]:
            if ("pin", 1) not in tried:
                return Action("pin", 1, "trial spread suggests scheduler noise", "heuristic")
    # Nothing rule-based left: try an untried kernel, else an untried ef.
    for kern in ACTION_SPACE["kernel"]:
        if kern != cfg["kernel"] and ("kernel", kern) not in tried:
            return Action("kernel", kern, "no rule fired; sweep the remaining kernels",
                          "heuristic")
    for v in ACTION_SPACE["ef"]:
        if v != cfg["ef"] and ("ef", v) not in tried:
            return Action("ef", v, "no rule fired; sweep ef", "heuristic")
    return Action("ef", cfg["ef"], "action space exhausted", "heuristic")


def plan_random(snap, hyps, history, cfg, rng) -> Action:
    knob = rng.choice(list(ACTION_SPACE))
    return Action(knob, rng.choice(ACTION_SPACE[knob]), "random control arm", "random")


def plan_grid(snap, hyps, history, cfg, rng) -> Action:
    """Systematic sweep in a fixed order: the other control arm."""
    order = [(k, v) for k in ACTION_SPACE for v in ACTION_SPACE[k]]
    tried = {(h["action"]["knob"], h["action"]["value"]) for h in history}
    for knob, value in order:
        if (knob, value) not in tried:
            return Action(knob, value, "grid sweep", "grid")
    return Action("ef", cfg["ef"], "grid exhausted", "grid")


LLM_SYSTEM = """You tune a C++ recommendation serving engine. Each turn you get \
one measured snapshot, a ranked diagnosis, and the history of everything already \
tried with its verified outcome.

Choose exactly ONE knob and ONE value from the action space you are given. You \
cannot write code, run commands, or change anything else. Your choice is \
validated against the enumerated space and discarded if it is not in it.

The objective is to minimise p99 latency subject to retrieve_recall staying at \
or above the floor. Trade recall headroom for latency when there is headroom; \
never propose something that has already been verified as a regression.

Be specific about the mechanism in your rationale: say which stage of the \
request you expect to get cheaper and why."""


KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
DEFAULT_MODEL = {"claude": "claude-opus-5", "gemini": "gemini-2.5-flash"}


def llm_key_env(provider: str) -> str | None:
    return os.environ.get(KEY_ENV.get(provider, ""), "") or None


def _build_payload(snap, hyps, history, cfg) -> dict:
    return {
        "action_space": ACTION_SPACE,
        "current_config": cfg,
        "snapshot": {k: snap[k] for k in (
            "p99_mean_us", "p99_cv", "p50_us", "p95_us", "qps_mean", "retrieve_p99_us",
            "score_p99_us", "feature_p99_us", "hops_mean", "retrieve_recall", "response_recall", "rss_mib",
            "items", "dim", "ef", "retrieve_k", "kernel")},
        "diagnosis": [asdict(h) for h in hyps],
        "history": [
            {"config": h["config"], "action": h["action"], "accepted": h["accepted"],
             "p99_mean_us": h["p99_mean_us"], "response_recall": h["response_recall"],
             "reason": h["reason"]}
            for h in history[-20:]
        ],
    }


class PlannedAction(BaseModel):
    knob: str
    value: str
    rationale: str
    expected_stage: str


# Both backends take the same payload and return the same dict, or raise. The
# loop, the action space and the verification gate are identical either way --
# which is the point of the interface: changing the provider must not be able to
# change what counts as an accepted result. At ~12 calls per run of roughly 4k
# input and 500 output tokens, the cost difference between them is cents, so
# this exists for portability and for the A/B, not for the bill.
def _propose_claude(payload: dict, model: str) -> dict:
    import anthropic

    resp = anthropic.Anthropic().messages.parse(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=LLM_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        output_format=PlannedAction,
    )
    return resp.parsed_output.model_dump()


def _propose_gemini(payload: dict, model: str) -> dict:
    from google import genai
    from google.genai import types

    resp = genai.Client().models.generate_content(
        model=model,
        contents=json.dumps(payload, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=LLM_SYSTEM,
            response_mime_type="application/json",
            response_schema=PlannedAction,
        ),
    )
    return resp.parsed.model_dump()


PROPOSERS = {"claude": _propose_claude, "gemini": _propose_gemini}


def plan_llm(snap, hyps, history, cfg, rng, model: str, provider: str) -> Action:
    if not llm_key_env(provider):
        return plan_heuristic(snap, hyps, history, cfg, rng)
    try:
        p = PROPOSERS[provider](_build_payload(snap, hyps, history, cfg), model)
    except ImportError as e:
        print(f"  [llm/{provider}] SDK not installed ({e}); using heuristic", file=sys.stderr)
        return plan_heuristic(snap, hyps, history, cfg, rng)
    except Exception as e:  # noqa: BLE001 - no API failure may stop the loop
        print(f"  [llm/{provider}] {type(e).__name__}: {e}; using heuristic", file=sys.stderr)
        return plan_heuristic(snap, hyps, history, cfg, rng)

    # The boundary: nothing the model returns reaches the engine unchecked.
    knob, raw = p.get("knob", ""), p.get("value", "")
    value = raw
    if knob in ACTION_SPACE and ACTION_SPACE[knob] and isinstance(ACTION_SPACE[knob][0], int):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            pass
    act = Action(knob, value,
                 f"{p.get('rationale', '')} [stage: {p.get('expected_stage', '?')}]",
                 f"llm:{provider}")
    if not act.valid():
        print(f"  [llm/{provider}] proposed {knob}={raw!r}, outside the action space; "
              f"discarded", file=sys.stderr)
        return plan_heuristic(snap, hyps, history, cfg, rng)
    return act


PLANNERS = {"heuristic": plan_heuristic, "random": plan_random, "grid": plan_grid}


# --------------------------------------------------------------------------
# 4-5. EXECUTE and VERIFY
# --------------------------------------------------------------------------
def calibrate(w: Workload, repeats: int) -> dict[str, float]:
    """Measure this host's own run-to-run noise before trusting any comparison.

    The trial-to-trial spread inside one process is NOT the uncertainty of
    comparing two separate benchmark processes: thermal state, page cache and
    scheduler placement all change between invocations. Measured here on the
    same config, n=600/5 trials gave a between-run CV of 5.7% against a
    within-run SEM far below that -- which is exactly how a no-op change got
    accepted as a 10% win.

    So the gate's sigma comes from repeating the SAME config `repeats` times and
    taking the standard deviation of the result. Everything the agent later
    claims is measured against this floor.
    """
    xs = [observe(DEFAULT_CONFIG, w)["p99_mean_us"] for _ in range(max(2, repeats))]
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return {"p99_mean_us": mean, "p99_sd_us": sd,
            "cv": sd / mean if mean else 0.0,
            "detect_floor_frac": 2.0 * math.sqrt(2.0) * sd / mean if mean else 0.0,
            "samples": xs}


@dataclass
class Gate:
    recall_floor: float
    rss_budget_mib: float
    sigma_us: float          # calibrated between-run standard deviation
    sigma_multiple: float = 2.0


def verify(best: dict, cand: dict, gate: Gate) -> tuple[bool, str]:
    """Accept only a real, non-regressing improvement."""
    # Gate on END-TO-END recall: what the service returned, not what the index
    # could have returned. Gating on retrieve_recall alone would let the agent
    # shrink retrieve_k for free, since that never reaches the index probe.
    if cand["response_recall"] < gate.recall_floor:
        return False, (f"response_recall {cand['response_recall']:.4f} < floor "
                       f"{gate.recall_floor:.4f}")
    if cand["rss_mib"] > gate.rss_budget_mib:
        return False, f"rss {cand['rss_mib']:.1f} MiB > budget {gate.rss_budget_mib:.1f}"
    # Combined uncertainty of the two independent p99 estimates.
    # Two independent runs, so the difference carries twice the calibrated
    # per-run variance. sigma is the measured between-run SD, not the
    # within-run SEM -- the latter is a property of one process and says
    # nothing about comparing two.
    noise = gate.sigma_multiple * math.sqrt(2.0) * gate.sigma_us
    delta = best["p99_mean_us"] - cand["p99_mean_us"]
    if delta <= noise:
        return False, (f"p99 delta {delta:+.1f} us within noise band "
                       f"+/-{noise:.1f} us ({gate.sigma_multiple} sigma)")
    return True, f"p99 {best['p99_mean_us']:.1f} -> {cand['p99_mean_us']:.1f} us ({delta:+.1f}, noise {noise:.1f})"


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def run_loop(planner: str, iterations: int, w: Workload, seed: int, model: str,
             recall_floor_frac: float, rss_slack: float, provider: str = "claude",
             confirm_repeats: int = 1, calibrate_repeats: int = 5,
             start_config: dict | None = None, verbose: bool = True) -> dict:
    rng = random.Random(seed)
    cfg = dict(DEFAULT_CONFIG)

    if start_config:
        cfg.update(start_config)
    cal = calibrate(w, calibrate_repeats)
    baseline = observe(cfg, w)
    gate = Gate(recall_floor=baseline["response_recall"] * recall_floor_frac,
                rss_budget_mib=baseline["rss_mib"] * rss_slack,
                sigma_us=cal["p99_sd_us"])
    best = baseline
    best_cfg = dict(cfg)
    history: list[dict] = []
    trace = [{"iteration": 0, "p99_mean_us": baseline["p99_mean_us"],
              "response_recall": baseline["response_recall"],
              "retrieve_recall": baseline["retrieve_recall"], "accepted": True,
              "config": dict(cfg)}]

    if verbose:
        print(f"\n=== planner={planner} ===")
        print(f"calibration  p99={cal['p99_mean_us']:.1f}us sd={cal['p99_sd_us']:.1f}us "
              f"cv={cal['cv'] * 100:.1f}%  -> smallest detectable change "
              f"{cal['detect_floor_frac'] * 100:.1f}%")
        print(f"baseline  p99={baseline['p99_mean_us']:.1f}us "
              f"recall={baseline['response_recall']:.4f} "
              f"rss={baseline['rss_mib']:.1f}MiB  "
              f"| floor={gate.recall_floor:.4f} budget={gate.rss_budget_mib:.1f}MiB")

    t0 = time.perf_counter()
    for it in range(1, iterations + 1):
        hyps = diagnose(best, gate.recall_floor)
        if planner == "llm":
            act = plan_llm(best, hyps, history, best_cfg, rng, model, provider)
        else:
            act = PLANNERS[planner](best, hyps, history, best_cfg, rng)

        if not act.valid():
            if verbose:
                print(f"  [{it:2d}] invalid action {act.knob}={act.value}; skipped")
            continue

        cand_cfg = dict(best_cfg)
        cand_cfg[act.knob] = act.value
        if cand_cfg == best_cfg:
            if verbose:
                print(f"  [{it:2d}] no-op ({act.knob} already {act.value}); skipped")
            history.append({"config": cand_cfg, "action": asdict(act), "accepted": False,
                            "p99_mean_us": best["p99_mean_us"],
                            "response_recall": best["response_recall"],
                        "retrieve_recall": best["retrieve_recall"],
                            "reason": "no-op"})
            continue

        # Paired measurement. Trial-to-trial CV inside one process understates
        # run-to-run drift (thermal state, other load): re-running the SAME
        # config minutes apart moved p99 by 34% here, far outside the within-run
        # band. So the incumbent is re-measured immediately before the
        # candidate, and the two are compared in the same time window.
        incumbent = observe(best_cfg, w)
        cand = observe(cand_cfg, w)
        ok, reason = verify(incumbent, cand, gate)

        # Replication. Across ~12 comparisons at 2 sigma, a couple of false
        # positives are expected by construction, and a no-op change did slip
        # through before this existed. A win must survive being measured again
        # from scratch; a result that does not replicate was drift.
        if ok and confirm_repeats > 0:
            for r_i in range(confirm_repeats):
                inc2 = observe(best_cfg, w)
                cand2 = observe(cand_cfg, w)
                ok2, reason2 = verify(inc2, cand2, gate)
                if not ok2:
                    ok = False
                    reason = f"failed replication {r_i + 1}: {reason2}"
                    break
                reason = f"{reason}; replicated ({reason2})"

        history.append({"config": cand_cfg, "action": asdict(act), "accepted": ok,
                        "p99_mean_us": cand["p99_mean_us"],
                        "response_recall": cand["response_recall"],
                        "retrieve_recall": cand["retrieve_recall"],
                        "incumbent_p99_us": incumbent["p99_mean_us"],
                        "rss_mib": cand["rss_mib"], "reason": reason})
        trace.append({"iteration": it, "p99_mean_us": cand["p99_mean_us"],
                      "incumbent_p99_us": incumbent["p99_mean_us"],
                      "response_recall": cand["response_recall"],
                      "retrieve_recall": cand["retrieve_recall"], "accepted": ok,
                      "config": dict(cand_cfg)})

        if ok:
            best, best_cfg = cand, cand_cfg
        else:
            best = incumbent  # keep the freshest estimate of the incumbent
        # else: roll back -- best_cfg is untouched, so the next plan starts from it.

        if verbose:
            mark = "ACCEPT" if ok else "reject"
            print(f"  [{it:2d}] {mark} {act.knob}={act.value:<8} "
                  f"p99={cand['p99_mean_us']:7.1f}us (vs {incumbent['p99_mean_us']:6.1f}) "
                  f"recall={cand['response_recall']:.4f} "
                  f"| {reason}")
            print(f"       why: {act.rationale}")

    # Final confirmation. Every intermediate number came from a different
    # moment, so the headline improvement is measured once more here, with the
    # default config and the found config back to back in the same time window.
    start_cfg = {**DEFAULT_CONFIG, **(start_config or {})}
    final_base = observe(start_cfg, w)
    final_best = observe(best_cfg, w)
    elapsed = time.perf_counter() - t0
    improvement = ((final_base["p99_mean_us"] - final_best["p99_mean_us"])
                   / final_base["p99_mean_us"])
    confirmed, confirm_reason = verify(final_base, final_best, gate)
    if best_cfg == start_cfg:
        # Nothing was changed, so the only honest improvement is zero. Reporting
        # the difference between two measurements of the same config would be
        # reporting noise as a result.
        improvement = 0.0
        confirmed, confirm_reason = True, "no change from start; improvement is 0 by definition" 
    result = {
        "planner": planner,
        "provider": provider if planner == "llm" else None,
        "iterations": iterations, "seed": seed,
        "baseline_p99_us": final_base["p99_mean_us"],
        "baseline_recall": final_base["response_recall"],
        "best_p99_us": final_best["p99_mean_us"],
        "best_recall": final_best["response_recall"],
        "first_baseline_p99_us": baseline["p99_mean_us"],
        "final_confirmed": confirmed,
        "final_confirm_reason": confirm_reason,
        "start_config": start_cfg,
        "best_config": best_cfg,
        "improvement_frac": improvement,
        "accepted": sum(1 for h in history if h["accepted"]),
        "rejected": sum(1 for h in history if not h["accepted"]),
        "recall_floor": gate.recall_floor,
        "rss_budget_mib": gate.rss_budget_mib,
        "calibration": cal,
        "detect_floor_frac": cal["detect_floor_frac"],
        "wall_s": elapsed,
        "history": history,
        "trace": trace,
    }
    if verbose:
        print(f"  final A/B  start={final_base['p99_mean_us']:.1f}us -> "
              f"best={final_best['p99_mean_us']:.1f}us ({improvement * 100:+.1f}%) "
              f"confirmed={confirmed}")
        print(f"  {confirm_reason}")
        print(f"  config={best_cfg} "
              f"accepted={result['accepted']} "
              f"rejected={result['rejected']} in {elapsed:.0f}s")
    return result


# --------------------------------------------------------------------------
# PR validation gate
# --------------------------------------------------------------------------
def validate_pr(w: Workload, max_ratio: float, baseline_path: Path) -> int:
    """Agentic change validation: measure, compare to a stored baseline, verdict."""
    snap = observe(DEFAULT_CONFIG, w)
    tests = subprocess.run([str(exe("recserve_tests"))], cwd=ROOT, capture_output=True,
                           text=True)
    verdict: dict[str, Any] = {
        "tests_passed": tests.returncode == 0,
        "p99_mean_us": snap["p99_mean_us"],
        "p99_cv": snap["p99_cv"],
        "retrieve_recall": snap["retrieve_recall"],
        "response_recall": snap["response_recall"],
        "rss_mib": snap["rss_mib"],
    }
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(snap, indent=2) + "\n")
        verdict["status"] = "baseline_written"
        print(json.dumps(verdict, indent=2))
        return 0

    base = json.loads(baseline_path.read_text())
    ratio = snap["p99_mean_us"] / base["p99_mean_us"]
    recall_drop = base["response_recall"] - snap["response_recall"]
    verdict.update({
        "baseline_p99_us": base["p99_mean_us"], "p99_ratio": ratio,
        "baseline_recall": base["response_recall"], "recall_drop": recall_drop,
    })
    reasons = []
    if not verdict["tests_passed"]:
        reasons.append("unit tests failed")
    if ratio > max_ratio:
        reasons.append(f"p99 regressed {ratio:.2f}x (limit {max_ratio:.2f}x)")
    if recall_drop > 0.01:
        reasons.append(f"recall dropped {recall_drop:.4f}")
    verdict["status"] = "pass" if not reasons else "fail"
    verdict["reasons"] = reasons
    print(json.dumps(verdict, indent=2))
    return 0 if not reasons else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["heuristic", "llm", "random", "grid"],
                    default="heuristic")
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--compare", action="store_true",
                    help="run every planner on the same budget and compare")
    ap.add_argument("--validate-pr", action="store_true")
    ap.add_argument("--provider", choices=["claude", "gemini"], default="claude",
                    help="LLM backend for --planner llm; the loop is identical either way")
    ap.add_argument("--model", default="", help="defaults per provider")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--items", type=int, default=65536)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=1024)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--recall-probe", type=int, default=96)
    ap.add_argument("--catalog", default="")
    ap.add_argument("--index", default="")
    ap.add_argument("--recall-floor-frac", type=float, default=0.98,
                    help="floor = this fraction of the baseline's recall")
    ap.add_argument("--rss-slack", type=float, default=1.25)
    ap.add_argument("--start-config", default="",
                    help="JSON overrides for the starting config, e.g. {\"ef\":256}. "
                         "Starting from a misconfigured system tests whether the loop can "
                         "recover, not just whether it avoids false positives.")
    ap.add_argument("--calibrate-repeats", type=int, default=5,
                    help="repeats of the default config used to measure this host's noise")
    ap.add_argument("--confirm-repeats", type=int, default=1,
                    help="paired re-measurements a win must also survive (0 disables)")
    ap.add_argument("--max-ratio", type=float, default=1.25, help="--validate-pr only")
    ap.add_argument("--baseline", default="results/agent_baseline.json")
    ap.add_argument("--out", default="results/agent_runs.jsonl")
    ap.add_argument("--summary", default="results/agent_summary.json")
    args = ap.parse_args()

    w = Workload(items=args.items, dim=args.dim, clusters=args.clusters, n=args.n,
                 trials=args.trials, recall_probe=args.recall_probe,
                 catalog=args.catalog, index=args.index)

    model = args.model or DEFAULT_MODEL[args.provider]
    start_config = json.loads(args.start_config) if args.start_config else None

    if args.validate_pr:
        return validate_pr(w, args.max_ratio, ROOT / args.baseline)

    if args.planner == "llm" and not llm_key_env(args.provider):
        print(f"note: {KEY_ENV[args.provider]} is unset; the llm planner falls back to "
              f"the heuristic planner on every turn", file=sys.stderr)

    planners = ["heuristic", "random", "grid", "llm"] if args.compare else [args.planner]
    results = []
    for p in planners:
        if p == "llm" and args.compare and not llm_key_env(args.provider):
            print(f"\n=== planner=llm === skipped: {KEY_ENV[args.provider]} unset")
            continue
        results.append(run_loop(p, args.iterations, w, args.seed, model,
                                args.recall_floor_frac, args.rss_slack, args.provider,
                                args.confirm_repeats, args.calibrate_repeats,
                                start_config))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in results:
            f.write(json.dumps({"ts": time.time(), **r}) + "\n")

    summary = {
        "workload": asdict(w),
        "runs": [{k: v for k, v in r.items() if k not in ("history", "trace")}
                 for r in results],
        "traces": {r["planner"]: r["trace"] for r in results},
    }
    (ROOT / args.summary).write_text(json.dumps(summary, indent=2) + "\n")

    if len(results) > 1:
        print(f"\n{'planner':10s} {'default_p99':>12s} {'best_p99':>9s} {'improve':>9s} "
              f"{'confirmed':>10s} {'acc':>4s} {'rej':>4s} {'recall':>8s}  best_config")
        for r in sorted(results, key=lambda x: -x["improvement_frac"]):
            print(f"{r['planner']:10s} {r['baseline_p99_us']:12.1f} {r['best_p99_us']:9.1f} "
                  f"{r['improvement_frac'] * 100:+8.1f}% {str(r['final_confirmed']):>10s} "
                  f"{r['accepted']:4d} {r['rejected']:4d} {r['best_recall']:8.4f}  "
                  f"{r['best_config']}")
    print(f"\n-> {out}\n-> {ROOT / args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
