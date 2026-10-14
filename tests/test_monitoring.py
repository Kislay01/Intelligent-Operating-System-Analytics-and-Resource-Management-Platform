import os
import sys
import time

import pytest

from kernelsense.monitoring import PsutilCollector, WorkloadSignal, build_tree, classify, detect_anomalies, render_tree
from kernelsense.monitoring import procfs
from kernelsense.monitoring.collector import ProcessInfo
from kernelsense.storage import MetricStore

linux_only = pytest.mark.skipif(not procfs.is_linux_procfs(), reason="needs Linux /proc")


def make(pid, ppid, state="S", name="p", **kw):
    base = dict(pid=pid, ppid=ppid, name=name, username="user", state=state, cpu_percent=0.0, rss_mb=1.0,
                vms_mb=1.0, mem_percent=0.0, num_threads=1, nice=0, vol_ctx=0, invol_ctx=0, minflt=0, majflt=0,
                read_bytes=0, write_bytes=0, num_fds=3, create_time=time.time(), cmdline="")
    base.update(kw)
    return ProcessInfo(**base)


def test_classification_rules():
    assert classify(WorkloadSignal(95, 2, 300, 0, 1)) == "CPU-bound"
    assert classify(WorkloadSignal(5, 400, 3, 5_000_000, 1)) == "I/O-bound"
    assert classify(WorkloadSignal(3, 15, 1, 0, 1)) == "Interactive"
    assert classify(WorkloadSignal(0.0, 0, 0, 0, 1)) == "Idle"


def test_tree_and_anomalies():
    procs = [make(1, 0, name="init"), make(10, 1, name="shell"), make(11, 10, "Z", name="child"),
             make(12, 10, "D", name="dd"), make(13, 1, name="orphan")]
    tree = build_tree(procs)
    assert tree[10] == [11, 12]
    text = render_tree(procs)
    assert "child (11) [Z]" in text
    kinds = {(a.kind, a.pid) for a in detect_anomalies(procs)}
    assert ("Zombie process", 11) in kinds
    assert ("Uninterruptible sleep", 12) in kinds
    assert ("Orphan process", 13) in kinds


def test_fork_storm_detected_but_not_for_kthreadd():
    procs = [make(2, 0, name="kthreadd")] + [make(100 + i, 2) for i in range(60)]
    procs += [make(500, 1, name="bomb")] + [make(600 + i, 500) for i in range(60)]
    storms = [a.pid for a in detect_anomalies(procs) if a.kind == "Possible fork storm"]
    assert storms == [500]


@linux_only
def test_procfs_readers():
    st = procfs.read_stat(os.getpid())
    assert st and st["state"] in "RSD" and st["ppid"] == os.getppid()
    assert procfs.context_switches(os.getpid()) is not None
    assert procfs.read_loadavg()["total_tasks"] > 0
    assert procfs.read_stat(10**9) is None


def test_collector_and_store(tmp_path):
    c = PsutilCollector()
    c.processes()
    time.sleep(0.3)
    s, procs = c.system(), c.processes()
    assert 0 <= s.mem_percent <= 100 and s.process_count > 0
    me = next(p for p in procs if p.pid == os.getpid())
    assert me.name.startswith("python") or "pytest" in me.cmdline or sys.executable
    store = MetricStore(tmp_path / "t.db")
    store.insert_system(s.to_row())
    from kernelsense.monitoring import processes_to_rows
    store.insert_processes(processes_to_rows(procs, s.ts))
    assert store.row_count() == 1
    assert len(store.process_history(minutes=5)) == len(procs)
    assert not store.system_history(5).empty
