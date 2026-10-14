"""Experiments behind the reported results.

Each function returns a JSON-serialisable dict; ``run_all`` writes them to
``benchmarks/results/`` and renders ``benchmarks/RESULTS.md``.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from ..config import PROJECT_ROOT, settings

WORKLOADS = PROJECT_ROOT / "workloads"


# --------------------------------------------------------------------------- #
# 1. Closed-loop allocation: interactive latency under CPU contention
# --------------------------------------------------------------------------- #
def _spawn_hogs(n: int) -> list[subprocess.Popen]:
    return [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]


def _cpu_seconds(pids) -> float:
    total = 0.0
    for p in pids:
        try:
            t = psutil.Process(p).cpu_times()
            total += t.user + t.system
        except psutil.Error:
            pass
    return total


def allocation_trial(policy: str, n_hogs: int, duration: float) -> dict:
    from ..allocator.cgroup import detect
    from ..allocator.controller import ClosedLoopController, ControllerConfig

    hogs = _spawn_hogs(n_hogs)
    pids = [h.pid for h in hogs]
    out_file = Path(tempfile.mkstemp(suffix=".json")[1])
    controller = None
    decisions = []
    try:
        time.sleep(0.3)
        if policy == "static nice 19":
            for p in pids:
                psutil.Process(p).nice(19)
        probe = subprocess.Popen([sys.executable, str(WORKLOADS / "latency_probe.py"), "--duration", str(duration),
                                  "--out", str(out_file)], stdout=subprocess.DEVNULL)
        cpu0 = _cpu_seconds(pids)
        if policy in ("rule-based", "Q-learning", "Q-learning + cgroup"):
            mode = "cgroup" if policy.endswith("cgroup") else "nice"
            if mode == "cgroup" and (detect() is None or not detect().writable):
                raise RuntimeError("cgroups not writable")
            cfg = ControllerConfig(policy="rule" if policy == "rule-based" else "q", mode=mode, interval=1.0)
            controller = ClosedLoopController(pids, [probe.pid], cfg)
            stop = time.time() + duration - 0.5
            while time.time() < stop and probe.poll() is None:
                decisions.append(controller.step())
        probe.wait(timeout=duration + 30)
        cpu1 = _cpu_seconds(pids)
        res = json.loads(out_file.read_text())
        res["batch_cpu_seconds"] = round(cpu1 - cpu0, 2)
        res["nice_chosen"] = sorted({d.nice for d in decisions}) if decisions else None
        res["rollbacks"] = sum(d.rolled_back for d in decisions)
        return res
    finally:
        if controller is not None:
            controller.close()
        for h in hogs:
            h.kill()
        out_file.unlink(missing_ok=True)


def _ensure_policy() -> None:
    from ..allocator.learning import QLearner, train
    from ..config import settings

    path = settings.data_dir / "q_policy.json"
    try:
        QLearner.load(path)
    except (OSError, ValueError, KeyError):
        _, q, _ = train()
        q.save(path)


def allocation_experiment(trials: int = 3, duration: float = 8.0, n_hogs: int | None = None,
                          policies: list[str] | None = None) -> dict:
    from ..allocator.cgroup import detect

    n_hogs = n_hogs or 2 * (os.cpu_count() or 1)
    info = detect()
    _ensure_policy()
    policies = policies or ["default (no control)", "static nice 19", "rule-based", "Q-learning"] + (
        ["Q-learning + cgroup"] if info is not None and info.writable else [])
    rows = []
    for t in range(trials):
        for pol in policies:
            r = allocation_trial(pol, n_hogs, duration)
            r.update(policy=pol, trial=t)
            rows.append(r)
    df = pd.DataFrame(rows)
    agg = df.groupby("policy", sort=False).agg(
        p50_ms=("p50_ms", "mean"), p95_ms=("p95_ms", "mean"), p99_ms=("p99_ms", "mean"),
        p99_std=("p99_ms", "std"), batch_cpu_s=("batch_cpu_seconds", "mean"),
    ).round(3).reset_index()
    base = agg.loc[agg.policy == "default (no control)"].iloc[0]
    agg["p99_reduction_%"] = ((base.p99_ms - agg.p99_ms) / base.p99_ms * 100).round(1)
    agg["batch_throughput_%"] = (agg.batch_cpu_s / base.batch_cpu_s * 100).round(1)
    return {"cpus": os.cpu_count(), "batch_hogs": n_hogs, "trials": trials, "duration_s": duration,
            "cgroup_version": info.version if info else None, "summary": agg.to_dict("records"),
            "raw": df.drop(columns=["pid"], errors="ignore").to_dict("records")}


def measure_wakeup_latency(nice_levels=(0, 5, 10, 15, 19), duration: float = 4.0) -> dict[int, float]:
    """Real p95 wake-up latency of the probe while 2 CPU hogs per CPU run at each nice level."""
    out = {}
    n_hogs = 2 * (os.cpu_count() or 1)
    for nice in nice_levels:
        hogs = _spawn_hogs(n_hogs)
        try:
            for h in hogs:
                psutil.Process(h.pid).nice(nice)
            time.sleep(0.3)
            r = subprocess.run([sys.executable, str(WORKLOADS / "latency_probe.py"), "--duration", str(duration)],
                               capture_output=True, text=True, check=False)
            out[nice] = json.loads(r.stdout)["p95_ms"]
        finally:
            for h in hogs:
                h.kill()
    return out


def policy_learning(calibrate_on_machine: bool = True) -> dict:
    from ..allocator import env
    from ..allocator.learning import evaluate, train
    from ..config import settings

    calib = None
    if calibrate_on_machine:
        measured = measure_wakeup_latency()
        calib = env.calibrate(measured, n_batch=2)
        calib["measured_p95_ms"] = measured
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "sim_calibration.json").write_text(json.dumps(calib))
    bandit, q, hist = train()
    table = evaluate(bandit, q)
    from ..config import settings
    try:
        q.save(settings.data_dir / "q_policy.json")
    except OSError:
        pass
    return {"calibration": calib, "episodes": len(hist), "evaluation": table.to_dict("records"),
            "policy_table": q.policy_table().to_dict("records"),
            "learning_curve": hist.rolling(200).mean().iloc[199::200][["bandit_reward", "q_reward"]].round(4)
            .to_dict("records")}


# --------------------------------------------------------------------------- #
# 2. Collector overhead
# --------------------------------------------------------------------------- #
def collector_overhead(samples: int = 20) -> dict:
    from ..monitoring import PsutilCollector
    from ..monitoring.procfs_collector import ProcfsCollector

    out = {"processes": len(psutil.pids())}
    for name, cls in (("psutil", PsutilCollector), ("procfs", ProcfsCollector)):
        c = cls()
        c.processes()
        times = []
        for _ in range(samples):
            t0 = time.perf_counter()
            c.system()
            c.processes()
            times.append((time.perf_counter() - t0) * 1000)
        out[f"{name}_ms_per_sample"] = round(statistics.median(times), 2)
    # CPU cost of the recorder thread at the default 2 s interval
    from ..recorder import Recorder
    from ..storage import MetricStore

    rec = Recorder(MetricStore(":memory:"), interval=2.0)
    me = psutil.Process()
    t0 = me.cpu_times()
    w0 = time.time()
    for _ in range(5):
        rec.sample_once()
        time.sleep(2.0)
    t1 = me.cpu_times()
    out["recorder_cpu_percent_at_2s"] = round(100 * ((t1.user + t1.system) - (t0.user + t0.system)) / (time.time() - w0), 2)
    out["speedup_procfs_vs_psutil"] = round(out["psutil_ms_per_sample"] / out["procfs_ms_per_sample"], 2)
    return out


# --------------------------------------------------------------------------- #
# 3. Memory forecasting
# --------------------------------------------------------------------------- #
def _memory_series(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    if kind == "leak":
        return 30 + rng.uniform(0.02, 0.08) * t + rng.normal(0, 0.4, n)
    if kind == "gc_sawtooth":
        period = rng.integers(30, 70)
        return 40 + rng.uniform(0.01, 0.05) * t + 6 * ((t % period) / period) + rng.normal(0, 0.3, n)
    if kind == "step_load":
        return 35 + 10 * (t > n * rng.uniform(0.3, 0.6)) + rng.normal(0, 0.5, n)
    if kind == "daily_cycle":
        return 45 + 5 * np.sin(2 * np.pi * t / rng.integers(80, 160)) + rng.normal(0, 0.4, n)
    return 40 + rng.normal(0, 0.6, n)  # stable


def forecasting_benchmark(series_per_kind: int = 8, n: int = 360, horizon: int = 30, seed: int = 0) -> dict:
    """Forecast the next ``horizon`` samples (60 s at a 2 s sampling interval) of each series.

    Besides every fixed model, "auto-selected" picks a model per series using only
    the training part (inner holdout), exactly as the dashboard does.
    """
    from ..memory.forecast import default_models, evaluate, metrics

    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("leak", "gc_sawtooth", "step_load", "daily_cycle", "stable"):
        for _ in range(series_per_kind):
            y = _memory_series(kind, n, rng)
            train, test = y[:-horizon], y[-horizon:]
            for m in default_models():
                pred = m.fit(train).predict(horizon).mean
                rows.append({"kind": kind, "model": m.name, **metrics(test, pred)})
            inner = evaluate(train, holdout=horizon / len(train))
            chosen = inner.dropna(subset=["MAE"]).iloc[0]["model"]
            model = next(m for m in default_models() if m.name == chosen)
            pred = model.fit(train).predict(horizon).mean
            rows.append({"kind": kind, "model": "Auto-selected", "chosen": chosen, **metrics(test, pred)})
    df = pd.DataFrame(rows)
    by_model = df.groupby("model").agg(MAE=("MAE", "mean"), MAPE=("MAPE", "mean")).sort_values("MAPE")
    naive = by_model.loc["Naive (last value)", "MAPE"]
    by_model["mape_reduction_vs_naive_%"] = ((naive - by_model.MAPE) / naive * 100).round(1)
    by_kind = df.pivot_table(index="kind", columns="model", values="MAPE", aggfunc="mean").round(3)
    leak = by_kind.loc["leak"]
    return {"series": int(len(df[df.model == "Auto-selected"])), "horizon_samples": horizon,
            "by_model": by_model.round(3).reset_index().to_dict("records"),
            "mape_by_kind": by_kind.reset_index().to_dict("records"),
            "leak_mape_reduction_best_vs_naive_%": round(float((leak["Naive (last value)"] - leak.min())
                                                               / leak["Naive (last value)"] * 100), 1)}


def leak_detection_benchmark(n_each: int = 40, seed: int = 0) -> dict:
    from ..memory.leak import detect_leaks

    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    ts = time.time() + np.arange(60) * 5.0
    pid = 1
    for kind in ("leak", "leak_sawtooth", "stable", "spiky", "step"):
        for _ in range(n_each):
            t = np.arange(60)
            if kind == "leak":
                rss = 100 + rng.uniform(0.2, 1.0) * t + rng.normal(0, 1, 60)
            elif kind == "leak_sawtooth":
                rss = 100 + rng.uniform(0.3, 1.0) * t - 10 * ((t % 12) == 11) + rng.normal(0, 1.5, 60)
            elif kind == "stable":
                rss = 200 + rng.normal(0, 2, 60)
            elif kind == "spiky":
                rss = 200 + rng.normal(0, 2, 60) + 40 * (rng.random(60) < 0.05)
            else:
                rss = 150 + 30 * (t > 30) + rng.normal(0, 1, 60)
            truth[pid] = kind.startswith("leak")
            rows += [{"ts": a, "pid": pid, "name": kind, "rss_mb": b} for a, b in zip(ts, rss)]
            pid += 1
    findings = detect_leaks(pd.DataFrame(rows))
    pred = {f.pid: f.suspected for f in findings}
    y = np.array([truth[p] for p in truth])
    yhat = np.array([pred.get(p, False) for p in truth])
    tp, fp, fn = int((y & yhat).sum()), int((~y & yhat).sum()), int((y & ~yhat).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"processes": len(truth), "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


def oom_score_validation() -> dict:
    from ..memory import oom

    df = oom.frame(500)
    if df.empty:
        return {"processes": 0}
    return {"processes": len(df), "exact_match_within_1": round(float((abs(df.oom_score - df.estimated_score) <= 1).mean()), 3)}


# --------------------------------------------------------------------------- #
# 4. Scheduling
# --------------------------------------------------------------------------- #
def scheduling_benchmark(seed: int = 0) -> dict:
    from ..scheduling import default_schedulers
    from ..scheduling.ml_burst import compare_predictors, summary, synthetic_sequences
    from ..scheduling.workload import PROFILES, synthetic

    rows = []
    for profile in PROFILES:
        for s in range(5):
            jobs = synthetic(profile, 10, seed + s)
            for sched in default_schedulers():
                m = sched.run(jobs).summary()
                rows.append({"profile": profile, **m})
    df = pd.DataFrame(rows)
    table = df.groupby("algorithm").agg(avg_waiting=("avg_waiting", "mean"), avg_response=("avg_response", "mean"),
                                        p95_response=("p95_response", "mean"), fairness=("fairness", "mean"),
                                        context_switches=("context_switches", "mean")).round(2)
    seqs = synthetic_sequences(seed=seed)
    burst = summary(compare_predictors(seqs[:90], seqs[90:]))
    return {"policies": table.sort_values("avg_waiting").reset_index().to_dict("records"),
            "burst_prediction": burst.to_dict("records")}


def trace_capture_demo(duration: float = 4.0) -> dict:
    from ..scheduling import recommend
    from ..scheduling.trace import capture

    worker = subprocess.Popen([sys.executable, "-c", (
        "import time\nend=time.time()+%f\nwhile time.time()<end:\n t=time.time()\n"
        " while time.time()-t<0.06: pass\n time.sleep(0.09)\n") % (duration + 1)])
    try:
        tr = capture(duration, 20)
    finally:
        worker.kill()
    seq = tr.sequences().get(worker.pid, [])
    jobs = tr.to_jobs()
    rec = recommend(jobs, "interactive", practical_only=True) if len(jobs) >= 3 else None
    return {"bursts": len(tr.bursts), "processes": len({b.pid for b in tr.bursts}),
            "probe_burst_ms_median": statistics.median(seq) if seq else None,
            "probe_expected_ms": 60, "replayed_jobs": len(jobs), "recommended": rec.best if rec else None}


# --------------------------------------------------------------------------- #
# 5. Deadlock / paging / logs
# --------------------------------------------------------------------------- #
def lockdep_benchmark(n_programs: int = 200) -> dict:
    from ..deadlock.seeded import benchmark

    short = benchmark(n_programs, hold_s=0.0)
    longer = benchmark(n_programs // 2, hold_s=0.002, seed=1)
    return {"programs": n_programs,
            "lockdep_detection_rate": short["lockdep_detection_rate"],
            "lockdep_false_positive_rate": short["lockdep_false_positive_rate"],
            "observed_deadlocks_short_sections": short["observed_deadlock_rate_buggy"],
            "observed_deadlocks_2ms_sections": longer["observed_deadlock_rate_buggy"]}


def paging_benchmark() -> dict:
    from ..paging import ALL_POLICIES, LRU, mattson_lru_curve
    from ..paging import traces

    rng = random.Random(0)
    refs = [rng.randrange(40) for _ in range(3000)]
    t0 = time.perf_counter()
    m = mattson_lru_curve(refs, 32).faults.tolist()
    t_m = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = [LRU().simulate(refs, c).faults for c in range(1, 33)]
    t_s = time.perf_counter() - t0
    scan = []
    for i in range(40):
        scan += [rng.randrange(5) for _ in range(20)] + list(range(1000 + i * 30, 1030 + i * 30))
    scan_res = {a: ALL_POLICIES[a]().simulate(scan, 8).faults for a in ("FIFO", "LRU", "Clock", "ARC", "Optimal")}
    algo = {}
    for name in traces.ALGORITHMIC:
        r = traces.algorithmic(name)
        algo[name] = {"refs": len(r), "pages": len(set(r)),
                      **{a: ALL_POLICIES[a]().simulate(r, 8).faults for a in ("FIFO", "LRU", "ARC", "Optimal")}}
    return {"mattson_matches_simulation": m == s, "mattson_seconds": round(t_m, 3),
            "per_size_simulation_seconds": round(t_s, 3), "mattson_speedup": round(t_s / t_m, 1),
            "scan_workload_faults_8_frames": scan_res, "algorithmic_traces_8_frames": algo}


def loghub_benchmark() -> dict:
    from ..logs import loghub

    try:
        return loghub.run_all()
    except OSError as exc:
        return {"error": f"Loghub datasets unavailable: {exc}"}


# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "collector_overhead": collector_overhead,
    "scheduling": scheduling_benchmark,
    "trace_capture": trace_capture_demo,
    "forecasting": forecasting_benchmark,
    "leak_detection": leak_detection_benchmark,
    "oom_score": oom_score_validation,
    "lockdep": lockdep_benchmark,
    "paging": paging_benchmark,
    "loghub": loghub_benchmark,
    "policy_learning": policy_learning,
    "allocation": allocation_experiment,
}


def run(names: list[str] | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names or list(EXPERIMENTS):
        t0 = time.time()
        print(f"[{name}] running...", flush=True)
        try:
            res = EXPERIMENTS[name]()
        except Exception as exc:  # noqa: BLE001 - record failures in the report
            res = {"error": f"{type(exc).__name__}: {exc}"}
        res["_seconds"] = round(time.time() - t0, 1)
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2, default=str))
        results[name] = res
        print(f"[{name}] done in {res['_seconds']}s", flush=True)
    return results
