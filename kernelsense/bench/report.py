"""Render benchmark JSON results into benchmarks/RESULTS.md."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd


def _table(rows, cols=None) -> str:
    if not rows:
        return "_no data_\n"
    df = pd.DataFrame(rows)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))
    return f"{head}\n{sep}\n{body}\n"


def _load(d: Path, name: str) -> dict:
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def render(results_dir: Path, out: Path) -> Path:
    r = {n: _load(results_dir, n) for n in (
        "collector_overhead", "scheduling", "trace_capture", "forecasting", "leak_detection", "oom_score",
        "lockdep", "paging", "loghub", "policy_learning", "allocation")}
    lines = [
        "# Benchmark results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} on {platform.system()} {platform.release()} "
        f"({platform.machine()}), Python {platform.python_version()}. "
        "Reproduce with `python benchmarks/run_all.py`; raw numbers are in `benchmarks/results/*.json`.",
        "",
    ]

    a = r["allocation"]
    if a.get("summary"):
        lines += ["## Closed-loop resource allocation",
                  "",
                  f"A latency probe (sleeps 4 ms, then does 1 ms of work) competes with {a['batch_hogs']} CPU-bound "
                  f"batch processes on {a['cpus']} CPUs. Wake-up latency = how late the probe runs after its timer "
                  f"fires. {a['trials']} trials x {a['duration_s']} s per policy; cgroup v{a.get('cgroup_version')}.",
                  "",
                  _table(a["summary"], ["policy", "p50_ms", "p95_ms", "p99_ms", "p99_std", "p99_reduction_%",
                                        "batch_throughput_%"]),
                  "`batch_throughput_%` = CPU time the batch processes received relative to no control.", ""]
    p = r["policy_learning"]
    if p.get("evaluation"):
        cal = p.get("calibration") or {}
        lines += ["## Policy learning in the calibrated simulator", ""]
        if cal:
            lines += [f"Simulator calibrated against measured wake-up latency (p95 ms by batch nice: "
                      f"{cal.get('measured_p95_ms')}) → preempt ratio {cal.get('preempt_ratio')}.", ""]
        lines += [f"{p['episodes']} training episodes; evaluation on 300 held-out scenarios "
                  "(reward = batch throughput vs. fair share minus SLO violation penalty; regret vs. the best fixed "
                  "action per scenario).", "", _table(p["evaluation"]), ""]

    c = r["collector_overhead"]
    if c:
        lines += ["## Collector overhead", "",
                  _table([{"processes": c.get("processes"), "psutil ms/sample": c.get("psutil_ms_per_sample"),
                           "/proc ms/sample": c.get("procfs_ms_per_sample"),
                           "speedup": c.get("speedup_procfs_vs_psutil"),
                           "recorder CPU % @2s": c.get("recorder_cpu_percent_at_2s")}]), ""]

    s = r["scheduling"]
    if s.get("policies"):
        lines += ["## CPU scheduling", "", "Averages over 5 synthetic profiles x 5 seeds (10 processes each).", "",
                  _table(s["policies"]), "", "### Next-burst prediction (held-out sequences)", "",
                  _table(s.get("burst_prediction")), ""]
    t = r["trace_capture"]
    if t:
        lines += ["### Live burst trace capture", "",
                  f"Captured {t.get('bursts')} CPU bursts from {t.get('processes')} processes via /proc jiffy counters; "
                  f"a probe with known 60 ms bursts was measured at a median of {t.get('probe_burst_ms_median')} ms. "
                  f"Replayed as {t.get('replayed_jobs')} jobs; recommended policy: {t.get('recommended')}.", ""]

    f = r["forecasting"]
    if f.get("by_model"):
        lines += ["## Memory forecasting", "", f"{f.get('series')} synthetic memory series (leak, GC sawtooth, step "
                  f"load, daily cycle, stable); forecast horizon {f.get('horizon_samples')} samples. "
                  "'Auto-selected' chooses a model per series from the training part only.", "",
                  _table(f["by_model"]), "", "MAPE by series type:", "", _table(f.get("mape_by_kind")),
                  f"On leaking processes the best model cuts MAPE by {f.get('leak_mape_reduction_best_vs_naive_%')}% "
                  "versus the naive forecast.", ""]
    lk = r["leak_detection"]
    if lk:
        lines += [f"**Leak detection** on {lk.get('processes')} simulated processes (linear and sawtooth leaks vs "
                  f"stable, spiky and step-change processes): precision {lk.get('precision')}, recall "
                  f"{lk.get('recall')}, F1 {lk.get('f1')}.", ""]
    o = r["oom_score"]
    if o.get("processes"):
        lines += [f"**OOM victim prediction**: recomputed `oom_score` matches the kernel's value (±1) for "
                  f"{o['exact_match_within_1'] * 100:.0f}% of {o['processes']} user processes.", ""]

    ld = r["lockdep"]
    if ld:
        lines += ["## Deadlock prediction (lock-order validation)", "",
                  f"{ld.get('programs')} generated multi-threaded programs, half with a seeded lock-order inversion.", "",
                  _table([{"lockdep detection rate": ld.get("lockdep_detection_rate"),
                           "false positive rate": ld.get("lockdep_false_positive_rate"),
                           "deadlock actually observed (short sections)": ld.get("observed_deadlocks_short_sections"),
                           "deadlock observed (2 ms sections)": ld.get("observed_deadlocks_2ms_sections")}]), ""]

    pg = r["paging"]
    if pg:
        lines += ["## Page replacement", "",
                  f"Mattson single-pass LRU curve matches per-size simulation: **{pg.get('mattson_matches_simulation')}** "
                  f"({pg.get('mattson_speedup')}x faster for 32 frame sizes).", "",
                  "Hot set + sequential scans, 8 frames (faults): " +
                  ", ".join(f"{k} {v}" for k, v in (pg.get("scan_workload_faults_8_frames") or {}).items()), "",
                  "Algorithmic traces, 8 frames:", "",
                  _table([{"trace": k, **v} for k, v in (pg.get("algorithmic_traces_8_frames") or {}).items()]), ""]

    lh = r["loghub"]
    if lh.get("parsing"):
        lines += ["## Log analysis on Loghub", "", "Drain grouping accuracy:", "",
                  _table([{"dataset": k, **v} for k, v in lh["parsing"].items()]), ""]
        b = lh.get("bgl_anomaly", {})
        if b:
            lines += [f"BGL anomaly detection ({b.get('windows')} windows of {b.get('window_lines')} lines, "
                      f"{b.get('test_windows')} test windows):", "",
                      _table([{"method": "n-gram (DeepLog rule)", **b["ngram_deeplog"]},
                              {"method": "Isolation Forest", **b["isolation_forest"]}]), ""]
    out.write_text("\n".join(lines))
    return out
