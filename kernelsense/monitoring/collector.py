"""Process and system collectors.

``Collector`` defines the interface the rest of the platform depends on. The
current implementation, ``PsutilCollector``, uses psutil for portability and
reads a few fields straight from /proc. A pure-/proc collector can be dropped
in later without changing any analytics code.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import psutil

from . import procfs
from .classify import WorkloadSignal, classify

PSUTIL_STATE_TO_LETTER = {
    psutil.STATUS_RUNNING: "R",
    psutil.STATUS_SLEEPING: "S",
    psutil.STATUS_DISK_SLEEP: "D",
    psutil.STATUS_STOPPED: "T",
    psutil.STATUS_TRACING_STOP: "t",
    psutil.STATUS_ZOMBIE: "Z",
    psutil.STATUS_DEAD: "X",
    getattr(psutil, "STATUS_IDLE", "idle"): "I",
    getattr(psutil, "STATUS_PARKED", "parked"): "P",
    getattr(psutil, "STATUS_WAKING", "waking"): "W",
    getattr(psutil, "STATUS_WAKE_KILL", "wake-kill"): "K",
}


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    name: str
    username: str
    state: str
    cpu_percent: float
    rss_mb: float
    vms_mb: float
    mem_percent: float
    num_threads: int
    nice: int
    vol_ctx: int
    invol_ctx: int
    minflt: int
    majflt: int
    read_bytes: int
    write_bytes: int
    num_fds: int
    create_time: float
    cmdline: str
    workload_class: str = "Unknown"
    vol_ctx_rate: float = 0.0
    invol_ctx_rate: float = 0.0
    io_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemSnapshot:
    ts: float
    cpu_percent: float
    per_cpu: list[float]
    mem_percent: float
    mem_total_mb: float
    mem_used_mb: float
    mem_available_mb: float
    mem_cached_mb: float
    swap_percent: float
    swap_used_mb: float
    load1: float
    load5: float
    load15: float
    runnable: int
    ctx_switches: int
    interrupts: int
    process_count: int
    thread_count: int
    boot_time: float
    pressure: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("per_cpu")
        d.pop("pressure")
        return d


class Collector(ABC):
    @abstractmethod
    def system(self) -> SystemSnapshot: ...

    @abstractmethod
    def processes(self) -> list[ProcessInfo]: ...


class PsutilCollector(Collector):
    """Collector based on psutil with /proc enrichment.

    Keeps psutil.Process objects alive between calls so ``cpu_percent`` is
    computed from the CPU-time delta since the previous sample (the same way
    top computes it from /proc/[pid]/stat jiffies).
    """

    ATTRS = [
        "pid", "ppid", "name", "username", "status", "memory_info", "memory_percent",
        "num_threads", "nice", "num_ctx_switches", "create_time", "cmdline",
    ]

    def __init__(self) -> None:
        self._procs: dict[int, psutil.Process] = {}
        self._prev: dict[int, tuple[float, int, int, int]] = {}  # pid -> (ts, vol, invol, io)
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

    # ---------------- system ----------------
    def system(self) -> SystemSnapshot:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        stats = psutil.cpu_stats()
        load = procfs.read_loadavg()
        if load is None:
            l1, l5, l15 = (os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0))
            load = {"load1": l1, "load5": l5, "load15": l15, "runnable": 0, "total_tasks": 0}
        thread_count = load.get("total_tasks", 0)
        return SystemSnapshot(
            ts=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            per_cpu=psutil.cpu_percent(interval=None, percpu=True),
            mem_percent=vm.percent,
            mem_total_mb=vm.total / 2**20,
            mem_used_mb=(vm.total - vm.available) / 2**20,
            mem_available_mb=vm.available / 2**20,
            mem_cached_mb=getattr(vm, "cached", 0) / 2**20,
            swap_percent=sw.percent,
            swap_used_mb=sw.used / 2**20,
            load1=load["load1"],
            load5=load["load5"],
            load15=load["load15"],
            runnable=load.get("runnable", 0),
            ctx_switches=stats.ctx_switches,
            interrupts=stats.interrupts,
            process_count=len(psutil.pids()),
            thread_count=thread_count,
            boot_time=psutil.boot_time(),
            pressure=procfs.pressure_snapshot(),
        )

    # ---------------- processes ----------------
    def _get_proc(self, pid: int) -> psutil.Process | None:
        proc = self._procs.get(pid)
        if proc is None:
            try:
                proc = psutil.Process(pid)
                proc.cpu_percent(interval=None)  # prime
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None
            self._procs[pid] = proc
        return proc

    def processes(self) -> list[ProcessInfo]:
        now = time.time()
        live = set(psutil.pids())
        for dead in set(self._procs) - live:
            self._procs.pop(dead, None)
            self._prev.pop(dead, None)

        result: list[ProcessInfo] = []
        for pid in live:
            proc = self._get_proc(pid)
            if proc is None:
                continue
            try:
                with proc.oneshot():
                    info = proc.as_dict(attrs=self.ATTRS, ad_value=None)
                    cpu = proc.cpu_percent(interval=None)
                    try:
                        io = proc.io_counters()
                        rb, wb = io.read_bytes, io.write_bytes
                    except (psutil.AccessDenied, AttributeError, psutil.NoSuchProcess):
                        rb = wb = 0
                    try:
                        fds = proc.num_fds() if hasattr(proc, "num_fds") else 0
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        fds = 0
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied:
                continue

            stat = procfs.read_stat(pid) or {}
            mem = info.get("memory_info")
            ctx = info.get("num_ctx_switches")
            vol = ctx.voluntary if ctx else 0
            invol = ctx.involuntary if ctx else 0
            io_total = rb + wb

            p = ProcessInfo(
                pid=pid,
                ppid=info.get("ppid") or 0,
                name=info.get("name") or "?",
                username=info.get("username") or "?",
                state=PSUTIL_STATE_TO_LETTER.get(info.get("status"), stat.get("state", "?")),
                cpu_percent=round(cpu or 0.0, 2),
                rss_mb=round(mem.rss / 2**20, 2) if mem else 0.0,
                vms_mb=round(mem.vms / 2**20, 2) if mem else 0.0,
                mem_percent=round(info.get("memory_percent") or 0.0, 3),
                num_threads=info.get("num_threads") or 0,
                nice=info.get("nice") if info.get("nice") is not None else 0,
                vol_ctx=vol,
                invol_ctx=invol,
                minflt=stat.get("minflt", 0),
                majflt=stat.get("majflt", 0),
                read_bytes=rb,
                write_bytes=wb,
                num_fds=fds,
                create_time=info.get("create_time") or 0.0,
                cmdline=" ".join(info.get("cmdline") or [])[:200],
            )

            prev = self._prev.get(pid)
            if prev:
                dt = max(now - prev[0], 1e-3)
                dv, di, dio = max(vol - prev[1], 0), max(invol - prev[2], 0), max(io_total - prev[3], 0)
                p.vol_ctx_rate = round(dv / dt, 2)
                p.invol_ctx_rate = round(di / dt, 2)
                p.io_rate = round(dio / dt, 1)
                p.workload_class = classify(WorkloadSignal(p.cpu_percent, dv, di, dio, dt))
            self._prev[pid] = (now, vol, invol, io_total)
            result.append(p)
        return result


def processes_to_rows(procs: list[ProcessInfo], ts: float) -> list[dict]:
    return [
        {
            "ts": ts,
            "pid": p.pid,
            "name": p.name,
            "state": p.state,
            "cpu_percent": p.cpu_percent,
            "rss_mb": p.rss_mb,
            "num_threads": p.num_threads,
            "vol_ctx": p.vol_ctx,
            "invol_ctx": p.invol_ctx,
            "read_bytes": p.read_bytes,
            "write_bytes": p.write_bytes,
            "nice": p.nice,
        }
        for p in procs
    ]
