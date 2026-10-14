import os
import subprocess
import sys

import pandas as pd
import psutil
import pytest

from kernelsense.allocator import Action, Actuator, PolicyConfig, SystemState, WorkloadClassifier, plan, synthetic_dataset


@pytest.fixture(scope="module")
def clf():
    c = WorkloadClassifier(n_estimators=60)
    c.train(synthetic_dataset(n_per_class=120))
    return c


def test_classifier_accuracy_and_importances(clf):
    assert clf.report.accuracy > 0.9
    assert set(clf.report.importances.index) >= {"cpu_mean", "rss_slope", "vol_rate"}


def test_classifier_on_canonical_profiles(clf):
    rows = pd.DataFrame([
        dict(pid=1, name="hog", cpu_mean=99, cpu_std=1, rss_mb=50, rss_slope=0, vol_rate=1, invol_rate=80,
             vol_ratio=0.01, io_rate_kb=0, threads=1),
        dict(pid=2, name="dd", cpu_mean=8, cpu_std=4, rss_mb=30, rss_slope=0, vol_rate=400, invol_rate=2,
             vol_ratio=0.99, io_rate_kb=40000, threads=1),
        dict(pid=3, name="leak", cpu_mean=0.2, cpu_std=0.1, rss_mb=90, rss_slope=150, vol_rate=1, invol_rate=0,
             vol_ratio=1.0, io_rate_kb=0, threads=1),
        dict(pid=4, name="sleep", cpu_mean=0, cpu_std=0, rss_mb=5, rss_slope=0, vol_rate=0.1, invol_rate=0,
             vol_ratio=1.0, io_rate_kb=0, threads=1),
    ])
    pred = clf.predict(rows).set_index("name")["predicted"].to_dict()
    assert pred == {"hog": "CPU-bound", "dd": "I/O-bound", "leak": "Memory-heavy", "sleep": "Idle"}


def test_policy_deprioritises_cpu_bound_under_load():
    classified = pd.DataFrame([
        dict(pid=4242, name="hog", predicted="CPU-bound", confidence=0.95, cpu_mean=99, rss_mb=10),
        dict(pid=4243, name="editor", predicted="Interactive", confidence=0.9, cpu_mean=3, rss_mb=100),
        dict(pid=1, name="systemd", predicted="CPU-bound", confidence=0.99, cpu_mean=99, rss_mb=10),
    ])
    actions = plan(classified, SystemState(95, 40, cpu_count=4), PolicyConfig(batch_nice=12))
    kinds = {(a.pid, a.kind, str(a.target)) for a in actions}
    assert (4242, "nice", "12") in kinds
    assert (4242, "affinity", "[2, 3]") in kinds
    assert all(a.pid != 1 for a in actions)


def test_low_confidence_is_skipped():
    classified = pd.DataFrame([dict(pid=50, name="x", predicted="CPU-bound", confidence=0.3, cpu_mean=90, rss_mb=1)])
    assert plan(classified, SystemState(95, 40, 2)) == []


def test_actuator_dry_run_apply_and_rollback(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        log = tmp_path / "actions.log"
        act = Action(child.pid, "sleeper", "nice", 0, 7, "test")
        dry = Actuator(dry_run=True, log_path=log, allow_pids={child.pid}).execute(act)
        assert dry.status == "dry-run" and psutil.Process(child.pid).nice() == 0

        real = Actuator(dry_run=False, log_path=log, allow_pids={child.pid})
        res = real.execute(act)
        assert res.status == "applied" and psutil.Process(child.pid).nice() == 7
        expected_entries = 2  # dry-run + applied
        if os.geteuid() == 0:  # lowering nice again needs root
            real.rollback([res])
            assert psutil.Process(child.pid).nice() == 0
            expected_entries = 3
        assert len(real.read_log()) == expected_entries
    finally:
        child.kill()


def test_actuator_refuses_system_and_unlisted_processes(tmp_path):
    act = Actuator(dry_run=False, log_path=tmp_path / "a.log", allow_pids={123456})
    assert act.execute(Action(1, "init", "nice", 0, 10, "x")).status == "refused"
    assert act.execute(Action(os.getpid(), "me", "nice", 0, 10, "x")).status == "refused"
    assert act.execute(Action(os.getpid(), "me", "sched_class", "A", "B", "x", applicable=False)).status == "advisory"
