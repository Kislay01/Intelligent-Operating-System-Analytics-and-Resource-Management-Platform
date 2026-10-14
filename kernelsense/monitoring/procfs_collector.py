"""Pure /proc collector - no psutil.

Everything is computed from raw kernel counters, the same way top/htop do it:

* process CPU%  = delta(utime + stime) [jiffies] / CLK_TCK / delta(wall time)
* system CPU%   = 1 - delta(idle + iowait) / delta(total) from /proc/stat
* memory        = MemTotal - MemAvailable from /proc/meminfo
* context switches from /proc/[pid]/status, faults and state from /proc/[pid]/stat

It implements the same ``Collector`` interface as ``PsutilCollector`` so every
module works unchanged. ``KS_COLLECTOR=procfs`` (the default on Linux) selects it.
"""

from __future__ import annotations

import os
import pwd
import time

from . import procfs
from .classify import WorkloadSignal, classify
from .collector import Collector, ProcessInfo, SystemSnapshot

PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024 if hasattr(os, "sysconf") else 4


def _cpu_busy(prev: list[int], cur: list[int]) -> float:
    """CPU utilisation between two /proc/stat samples (user nice system idle iowait irq softirq steal)."""
    d = [c - p for c, p in zip(cur, prev)]
    total = sum(d[:8])
    idle = d[3] + (d[4] if len(d) > 4 else 0)
    return 100.0 * (total - idle) / total if total > 0 else 0.0


class ProcfsCollector(Collector):
    def __init__(self, with_smaps: bool = False) -> None:
        if not procfs.is_linux_procfs():
            raise RuntimeError("ProcfsCollector requires Linux /proc")
        self.hz = procfs.clock_ticks()
        self.with_smaps = with_smaps
        counters = procfs.read_proc_stat_counters()
        self.boot_time = float(counters.get("btime", time.time()))
        self._prev_cpu = procfs.read_cpu_times()
        self._prev_proc: dict[int, tuple[float, int, int, int, int]] = {}  # pid -> (ts, ticks, vol, invol, io)
        self._users: dict[int, str] = {}
        self._mem_total_kb = procfs.read_meminfo().get("MemTotal", 1)

    # ------------------------------------------------------------------ helpers
    def _user(self, uid: int) -> str:
        if uid not in self._users:
            try:
                self._users[uid] = pwd.getpwuid(uid).pw_name
            except KeyError:
                self._users[uid] = str(uid)
        return self._users[uid]

    @staticmethod
    def _read(path: str) -> str | None:
        try:
            with open(path, "rb") as fh:
                return fh.read().decode(errors="replace")
        except OSError:
            return None

    def _io(self, pid: int) -> tuple[int, int]:
        text = self._read(f"/proc/{pid}/io")
        if not text:
            return 0, 0
        rb = wb = 0
        for line in text.splitlines():
            if line.startswith("read_bytes:"):
                rb = int(line.split()[1])
            elif line.startswith("write_bytes:"):
                wb = int(line.split()[1])
        return rb, wb

    @staticmethod
    def _fd_count(pid: int) -> int:
        try:
            return len(os.listdir(f"/proc/{pid}/fd"))
        except OSError:
            return 0

    # ------------------------------------------------------------------ system
    def system(self) -> SystemSnapshot:
        cpu = procfs.read_cpu_times()
        total = _cpu_busy(self._prev_cpu.get("cpu", cpu["cpu"]), cpu["cpu"]) if "cpu" in cpu else 0.0
        per = [
            round(_cpu_busy(self._prev_cpu.get(k, v), v), 1)
            for k, v in sorted(cpu.items(), key=lambda kv: (len(kv[0]), kv[0]))
            if k != "cpu"
        ]
        self._prev_cpu = cpu
        mem = procfs.read_meminfo()
        total_kb = mem.get("MemTotal", 1)
        avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
        swap_total = mem.get("SwapTotal", 0)
        swap_used = swap_total - mem.get("SwapFree", 0)
        load = procfs.read_loadavg() or {"load1": 0.0, "load5": 0.0, "load15": 0.0, "runnable": 0, "total_tasks": 0}
        counters = procfs.read_proc_stat_counters()
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
        return SystemSnapshot(
            ts=time.time(),
            cpu_percent=round(total, 1),
            per_cpu=per,
            mem_percent=round(100 * (total_kb - avail_kb) / total_kb, 1),
            mem_total_mb=total_kb / 1024,
            mem_used_mb=(total_kb - avail_kb) / 1024,
            mem_available_mb=avail_kb / 1024,
            mem_cached_mb=(mem.get("Cached", 0) + mem.get("Buffers", 0)) / 1024,
            swap_percent=round(100 * swap_used / swap_total, 1) if swap_total else 0.0,
            swap_used_mb=swap_used / 1024,
            load1=load["load1"],
            load5=load["load5"],
            load15=load["load15"],
            runnable=load.get("runnable", 0),
            ctx_switches=counters.get("ctxt", 0),
            interrupts=counters.get("intr", 0),
            process_count=len(pids),
            thread_count=load.get("total_tasks", 0),
            boot_time=self.boot_time,
            pressure=procfs.pressure_snapshot(),
        )

    # --------------------------------------------------------------- processes
    def processes(self) -> list[ProcessInfo]:
        now = time.time()
        out: list[ProcessInfo] = []
        seen = set()
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            st = procfs.read_stat(pid)
            status = procfs.read_status(pid)
            if not st or not status:
                continue
            seen.add(pid)
            ticks = st["utime_ticks"] + st["stime_ticks"]
            try:
                vol = int(status.get("voluntary_ctxt_switches", 0))
                invol = int(status.get("nonvoluntary_ctxt_switches", 0))
                uid = int(status.get("Uid", "0").split()[0])
            except ValueError:
                vol = invol = uid = 0
            rb, wb = self._io(pid)
            rss_kb = st["rss_pages"] * PAGE_KB
            if self.with_smaps:
                sm = procfs.read_smaps_rollup(pid)
                if sm:
                    rss_kb = sm["rss_kb"]
            raw_cmd = self._read(f"/proc/{pid}/cmdline") or ""
            cmdline = raw_cmd.replace("\x00", " ").strip()[:200]

            cpu = 0.0
            p = ProcessInfo(
                pid=pid, ppid=st["ppid"], name=st["comm"], username=self._user(uid), state=st["state"],
                cpu_percent=0.0, rss_mb=round(rss_kb / 1024, 2), vms_mb=round(st["vsize"] / 2**20, 2),
                mem_percent=round(100 * rss_kb / self._mem_total_kb, 3), num_threads=st["num_threads"],
                nice=st["nice"], vol_ctx=vol, invol_ctx=invol, minflt=st["minflt"], majflt=st["majflt"],
                read_bytes=rb, write_bytes=wb, num_fds=self._fd_count(pid),
                create_time=self.boot_time + st["starttime_ticks"] / self.hz, cmdline=cmdline,
            )
            prev = self._prev_proc.get(pid)
            if prev:
                dt = max(now - prev[0], 1e-3)
                cpu = 100.0 * (ticks - prev[1]) / self.hz / dt
                dv, di, dio = max(vol - prev[2], 0), max(invol - prev[3], 0), max(rb + wb - prev[4], 0)
                p.cpu_percent = round(max(cpu, 0.0), 2)
                p.vol_ctx_rate = round(dv / dt, 2)
                p.invol_ctx_rate = round(di / dt, 2)
                p.io_rate = round(dio / dt, 1)
                p.workload_class = classify(WorkloadSignal(p.cpu_percent, dv, di, dio, dt))
            self._prev_proc[pid] = (now, ticks, vol, invol, rb + wb)
            out.append(p)
        for dead in set(self._prev_proc) - seen:
            self._prev_proc.pop(dead, None)
        return out


def make_collector(kind: str | None = None) -> Collector:
    """Return the configured collector ('procfs' on Linux by default, else psutil)."""
    from .collector import PsutilCollector

    kind = (kind or os.environ.get("KS_COLLECTOR", "auto")).lower()
    if kind in ("procfs", "auto") and procfs.is_linux_procfs():
        try:
            return ProcfsCollector()
        except (RuntimeError, OSError):
            pass
    return PsutilCollector()
