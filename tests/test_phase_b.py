"""Tests for the kernel data layer, learning components and evaluation tooling."""

import os
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from kernelsense.monitoring import procfs

linux = pytest.mark.skipif(not procfs.is_linux_procfs(), reason="needs Linux /proc")


# ------------------------------------------------------------- data layer
@linux
def test_procfs_collector_matches_psutil_for_a_cpu_hog():
    from kernelsense.monitoring import PsutilCollector
    from kernelsense.monitoring.procfs_collector import ProcfsCollector

    hog = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        a, b = ProcfsCollector(), PsutilCollector()
        a.processes(), b.processes()
        time.sleep(1.0)
        pa = {p.pid: p for p in a.processes()}
        pb = {p.pid: p for p in b.processes()}
        assert abs(pa[hog.pid].cpu_percent - pb[hog.pid].cpu_percent) < 15
        assert pa[hog.pid].workload_class == "CPU-bound"
        assert abs(pa[hog.pid].create_time - pb[hog.pid].create_time) < 1.0
        s = a.system()
        assert 0 <= s.mem_percent <= 100 and len(s.per_cpu) == (os.cpu_count() or 1)
    finally:
        hog.kill()


@linux
def test_extended_readers():
    sm = procfs.read_smaps_rollup(os.getpid())
    assert sm is None or sm["pss_kb"] <= sm["rss_kb"]
    assert "pgfault" in procfs.read_vmstat(procfs.VMSTAT_KEYS)
    assert procfs.read_meminfo()["MemTotal"] > 0
    assert "cpu" in procfs.read_cpu_times()


@linux
def test_oom_score_formula_matches_kernel():
    from kernelsense.memory import oom

    df = oom.frame(50)
    if df.empty:
        pytest.skip("no oom_score available")
    assert (abs(df.oom_score - df.estimated_score) <= 1).mean() >= 0.8


# ------------------------------------------------------------- scheduling
def test_cfs_shares_follow_nice_weights():
    from kernelsense.scheduling import CFS, Job, cpu_share, nice_weight

    jobs = [Job("A", 0, 400, 0), Job("B", 0, 400, 5)]
    share = cpu_share(CFS().run(jobs), 0, 300)
    expected = nice_weight(0) / (nice_weight(0) + nice_weight(5))
    assert abs(share["A"] - expected) < 0.03


def test_ml_burst_predictor_beats_exponential_average():
    from kernelsense.scheduling.ml_burst import compare_predictors, summary, synthetic_sequences

    seqs = synthetic_sequences(n=80)
    s = summary(compare_predictors(seqs[:60], seqs[60:])).set_index("method")
    assert s.loc["gradient_boosting", "MAE"] < s.loc["exp_avg_a=0.5", "MAE"]


@linux
def test_trace_capture_and_replay(tmp_path):
    from kernelsense.scheduling.trace import BurstTrace, capture

    worker = subprocess.Popen([sys.executable, "-c",
                               "import time\nend=time.time()+3\nwhile time.time()<end:\n t=time.time()\n"
                               " while time.time()-t<0.05: pass\n time.sleep(0.1)\n"])
    try:
        tr = capture(2.0, 20)
    finally:
        worker.kill()
    seq = tr.sequences().get(worker.pid, [])
    assert len(seq) >= 3
    assert 20 <= np.median(seq) <= 120
    path = tr.save(tmp_path / "t.json")
    assert len(BurstTrace.load(path).bursts) == len(tr.bursts)
    assert tr.to_jobs()


# ------------------------------------------------------------------ memory
def test_quantile_gbm_and_coverage():
    from kernelsense.memory import QuantileGBMForecaster, interval_coverage

    rng = np.random.default_rng(0)
    y = 30 + 0.05 * np.arange(300) + rng.normal(0, 0.4, 300)
    fc = QuantileGBMForecaster().fit(y[:250]).predict(50)
    assert len(fc.mean) == 50 and np.all(fc.lower <= fc.upper)
    assert interval_coverage(y[250:], fc) >= 0.5


def test_mann_kendall_and_step_rejection():
    from kernelsense.memory import detect_leaks, mann_kendall

    rng = np.random.default_rng(2)
    assert mann_kendall(np.arange(40) + rng.normal(0, 1, 40))[2] < 1e-6
    assert mann_kendall(rng.normal(0, 1, 40))[2] > 0.01
    ts = time.time() + np.arange(60) * 5.0
    t = np.arange(60)
    df = pd.DataFrame({"ts": np.r_[ts, ts], "pid": [1] * 60 + [2] * 60, "name": ["step"] * 60 + ["leak"] * 60,
                       "rss_mb": np.r_[150 + 30 * (t > 30) + rng.normal(0, 1, 60), 100 + 0.8 * t + rng.normal(0, 1, 60)]})
    res = {f.name: f.suspected for f in detect_leaks(df)}
    assert res == {"step": False, "leak": True}


# ----------------------------------------------------------------- deadlock
def test_lockdep_detects_inversion_without_deadlock():
    from kernelsense.deadlock.lockdep import demo_dining_philosophers, demo_inversion

    assert len(demo_inversion().violations) == 1
    assert demo_dining_philosophers().violations
    assert not demo_dining_philosophers(ordered=True).violations


def test_lockdep_patch_threading_restores_lock():
    from kernelsense.deadlock import LockOrderMonitor, patch_threading

    real = threading.Lock
    mon = LockOrderMonitor()
    with patch_threading(mon):
        a = threading.Lock()
        b = threading.Lock()
        with a, b:
            pass
        with b, a:
            pass
    assert threading.Lock is real
    assert mon.violations


def test_seeded_benchmark():
    from kernelsense.deadlock.seeded import benchmark

    r = benchmark(30)
    assert r["lockdep_detection_rate"] == 1.0 and r["lockdep_false_positive_rate"] == 0.0


# ------------------------------------------------------------------- paging
def test_arc_is_scan_resistant():
    import random

    from kernelsense.paging import ARC, LRU

    rng = random.Random(0)
    refs = []
    for i in range(30):
        refs += [rng.randrange(5) for _ in range(20)] + list(range(100 + i * 30, 130 + i * 30))
    assert ARC().simulate(refs, 8).faults < LRU().simulate(refs, 8).faults


def test_algorithmic_traces_show_locality_effects():
    from kernelsense.paging import LRU
    from kernelsense.paging.traces import algorithmic, parse_lackey

    row, col = algorithmic("matrix_row_major"), algorithmic("matrix_col_major")
    assert LRU().simulate(row, 8).faults < LRU().simulate(col, 8).faults
    naive, tiled = algorithmic("matmul_naive"), algorithmic("matmul_blocked")
    assert LRU().simulate(tiled, 8).faults < LRU().simulate(naive, 8).faults
    sample = "==1== Lackey\nI  04016c30,3\n L 00001000,8\n S 00001008,8\n M 00003000,4\n"
    assert parse_lackey(sample) == [1, 3]


# --------------------------------------------------------------------- logs
def test_drain_groups_templates():
    from kernelsense.logs import Drain, grouping_accuracy

    lines = [f"Accepted password for user{i} from 10.0.0.{i} port {1000 + i}" for i in range(20)]
    lines += [f"Connection closed by 10.0.1.{i}" for i in range(10)]
    d = Drain()
    d.fit(lines)
    assert len(d.clusters) == 2
    assert grouping_accuracy(d.final_assignments(), [0] * 20 + [1] * 10) == 1.0


def test_sequence_model_flags_unusual_order():
    from kernelsense.logs import NGramSequenceModel

    normal = [[1, 2, 3, 4] * 5 for _ in range(10)]
    m = NGramSequenceModel(history=2, top_k=1).fit(normal)
    assert not m.is_anomalous([1, 2, 3, 4, 1, 2, 3, 4])
    assert m.is_anomalous([1, 3, 2, 4])


def test_incident_correlation():
    from kernelsense import logs

    text = "Oct 14 10:10:00 host kernel: Out of memory: Killed process 42 (java) total-vm:1kB"
    df = logs.parse_text(text, year=2026)
    times = pd.date_range("2026-10-14 10:00", "2026-10-14 10:20", freq="10s")
    mem = np.full(len(times), 40.0)
    k = times.searchsorted(pd.Timestamp("2026-10-14 10:10"))
    mem[k - 3:k] = [60, 80, 97]
    hist = pd.DataFrame({"time": times, "cpu_percent": 10.0, "mem_percent": mem, "swap_percent": 0.0, "load1": 1.0})
    res = logs.incident_timeline(hist, df)
    assert res["explained"] == 1
    assert "Memory" in res["correlated"].iloc[0]["related_metrics"]


# --------------------------------------------------------------- allocation
def test_simulator_tradeoff_and_learning():
    from kernelsense.allocator.env import ACTIONS, Scenario, simulate
    from kernelsense.allocator.learning import evaluate, train

    sc = Scenario(n_batch=2, burst=1, period=5)
    lat = [simulate(sc, a, 1500).latency_p95 for a in ACTIONS]
    assert lat[0] >= lat[-1]
    bandit, q, _ = train(600, horizon=400)
    table = evaluate(bandit, q, n=60, horizon=400).set_index("policy")
    assert table.loc["Q-learning", "reward"] > table.loc["default (nice 0)", "reward"]


def test_qlearner_roundtrip(tmp_path):
    from kernelsense.allocator.learning import QLearner

    q = QLearner()
    q.update((1, 2, 0), 3, 0.5)
    p = q.save(tmp_path / "q.json")
    assert QLearner.load(p).q == q.q


def test_cgroup_dry_run_commands():
    from kernelsense.allocator.cgroup import Cgroup, CgroupInfo

    from pathlib import Path

    v2 = Cgroup("demo", CgroupInfo(2, Path("/sys/fs/cgroup"), ["cpu", "memory"], True), dry_run=True)
    v2.set_cpu_weight(50)
    v2.set_cpu_max(25)
    v2.set_memory_high(512 * 2**20)
    assert any(l.endswith("cpu.weight <- 50") for l in v2.log)
    assert any(l.endswith("cpu.max <- 25000 100000") for l in v2.log)
    v1 = Cgroup("demo", CgroupInfo(1, Path("/sys/fs/cgroup"), ["cpu", "memory"], True), dry_run=True)
    v1.set_cpu_weight(50)
    assert any(l.endswith("cpu.shares <- 512") for l in v1.log)


# ----------------------------------------------------------------- exporter
def test_prometheus_exposition_is_grouped():
    from kernelsense.exporter import KernelSenseExporter

    text = KernelSenseExporter().render()
    names = [l.split("{")[0].split(" ")[0] for l in text.splitlines() if l and not l.startswith("#")]
    seen = []
    for n in names:
        if seen and seen[-1] == n:
            continue
        assert n not in seen
        seen.append(n)
    assert "kernelsense_cpu_utilization_percent" in seen


def test_ebpf_histogram_parser():
    from kernelsense import ebpf

    rows = ebpf.parse_hist("@usecs:\n[0]   5 |@|\n[2, 4)   10 |@@|\n[1K, 2K)  1 | |\n")
    assert [r["count"] for r in rows] == [5, 10, 1]
    assert ebpf.percentile_from_hist(rows, 50) == 4
