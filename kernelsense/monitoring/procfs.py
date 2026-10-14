"""Direct readers for Linux /proc interfaces.

psutil covers most fields, but a few kernel views are read here directly so the
platform shows exactly what the kernel reports:

* /proc/[pid]/status  - context switches, VmRSS, VmSwap, thread count
* /proc/[pid]/stat    - process state letter, minor/major page faults
* /proc/loadavg       - load averages and runnable/total task counts
* /proc/pressure/*    - Pressure Stall Information (Linux >= 4.20, if enabled)

All readers return ``None`` (or empty data) when the file is unavailable, so the
platform degrades gracefully on kernels or platforms that lack them.
"""

from __future__ import annotations

import os
from pathlib import Path

PROC = Path("/proc")

STATE_NAMES = {
    "R": "Running",
    "S": "Sleeping (interruptible)",
    "D": "Disk sleep (uninterruptible)",
    "Z": "Zombie",
    "T": "Stopped",
    "t": "Tracing stop",
    "X": "Dead",
    "I": "Idle (kernel thread)",
    "P": "Parked",
    "W": "Waking",
    "K": "Wakekill",
}


def is_linux_procfs() -> bool:
    return (PROC / "self" / "status").exists()


def read_status(pid: int) -> dict[str, str] | None:
    """Parse /proc/[pid]/status into a dict of raw string values."""
    try:
        text = (PROC / str(pid) / "status").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def read_stat(pid: int) -> dict | None:
    """Parse /proc/[pid]/stat.

    The comm field is wrapped in parentheses and may contain spaces, so we split
    on the last ')' before splitting the remaining fields.
    """
    try:
        raw = (PROC / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    try:
        rparen = raw.rindex(")")
        comm = raw[raw.index("(") + 1 : rparen]
        fields = raw[rparen + 2 :].split()
        # fields[0] is field 3 (state) in proc(5) numbering
        return {
            "comm": comm,
            "state": fields[0],
            "ppid": int(fields[1]),
            "minflt": int(fields[7]),
            "majflt": int(fields[9]),
            "utime_ticks": int(fields[11]),
            "stime_ticks": int(fields[12]),
            "priority": int(fields[15]),
            "nice": int(fields[16]),
            "num_threads": int(fields[17]),
            "starttime_ticks": int(fields[19]),
            "vsize": int(fields[20]),
            "rss_pages": int(fields[21]),
            "processor": int(fields[36]) if len(fields) > 36 else -1,
        }
    except (ValueError, IndexError):
        return None


def context_switches(pid: int) -> tuple[int, int] | None:
    status = read_status(pid)
    if not status:
        return None
    try:
        return int(status["voluntary_ctxt_switches"]), int(status["nonvoluntary_ctxt_switches"])
    except (KeyError, ValueError):
        return None


def read_loadavg() -> dict | None:
    try:
        parts = (PROC / "loadavg").read_text().split()
        running, total = parts[3].split("/")
        return {
            "load1": float(parts[0]),
            "load5": float(parts[1]),
            "load15": float(parts[2]),
            "runnable": int(running),
            "total_tasks": int(total),
            "last_pid": int(parts[4]),
        }
    except (OSError, ValueError, IndexError):
        return None


def read_pressure(resource: str) -> dict | None:
    """Read PSI for 'cpu', 'memory' or 'io'.

    Returns {'some': {'avg10':..,'avg60':..,'avg300':..,'total':..}, 'full': {...}}.
    """
    path = PROC / "pressure" / resource
    try:
        text = path.read_text()
    except OSError:
        return None
    result: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        kind, *pairs = line.split()
        values = {}
        for pair in pairs:
            k, _, v = pair.partition("=")
            try:
                values[k] = float(v)
            except ValueError:
                continue
        result[kind] = values
    return result or None


def pressure_snapshot() -> dict[str, dict | None]:
    return {res: read_pressure(res) for res in ("cpu", "memory", "io")}


def read_smaps_rollup(pid: int) -> dict[str, int] | None:
    """RSS / PSS / USS (kB) from /proc/[pid]/smaps_rollup.

    PSS divides shared pages among the processes sharing them; USS (private
    clean + private dirty) is the memory freed if the process exits.
    """
    try:
        text = (PROC / str(pid) / "smaps_rollup").read_text()
    except OSError:
        return None
    vals: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            vals[key.strip()] = int(parts[0])
    if not vals:
        return None
    return {
        "rss_kb": vals.get("Rss", 0),
        "pss_kb": vals.get("Pss", 0),
        "uss_kb": vals.get("Private_Clean", 0) + vals.get("Private_Dirty", 0),
        "swap_kb": vals.get("Swap", 0),
    }


def read_vmstat(keys: tuple[str, ...] | None = None) -> dict[str, int]:
    """Kernel VM counters (page faults, swap in/out, reclaim, OOM kills)."""
    out: dict[str, int] = {}
    try:
        for line in (PROC / "vmstat").read_text().splitlines():
            k, _, v = line.partition(" ")
            if keys is None or k in keys:
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


VMSTAT_KEYS = ("pgfault", "pgmajfault", "pswpin", "pswpout", "pgscan_kswapd", "pgsteal_kswapd", "oom_kill")


def read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in (PROC / "meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[k] = int(parts[0])  # kB
    except (OSError, ValueError):
        pass
    return out


def read_cpu_times() -> dict[str, list[int]]:
    """Per-CPU jiffy counters from /proc/stat ('cpu' = aggregate)."""
    out: dict[str, list[int]] = {}
    try:
        for line in (PROC / "stat").read_text().splitlines():
            if line.startswith("cpu"):
                name, *vals = line.split()
                out[name] = [int(v) for v in vals]
    except (OSError, ValueError):
        pass
    return out


def read_proc_stat_counters() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in (PROC / "stat").read_text().splitlines():
            parts = line.split()
            if parts[0] in ("ctxt", "btime", "processes", "procs_running", "procs_blocked"):
                out[parts[0]] = int(parts[1])
            elif parts[0] == "intr":
                out["intr"] = int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return out


def read_oom_score(pid: int) -> tuple[int, int] | None:
    """(oom_score, oom_score_adj). Higher oom_score = killed first by the OOM killer."""
    try:
        score = int((PROC / str(pid) / "oom_score").read_text())
        adj = int((PROC / str(pid) / "oom_score_adj").read_text())
        return score, adj
    except (OSError, ValueError):
        return None


def clock_ticks() -> int:
    try:
        return os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return 100


def capabilities() -> dict[str, bool]:
    """Probe which kernel interfaces are available on this machine."""
    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").exists() or Path(
        "/sys/fs/cgroup/unified/cgroup.controllers").exists()
    return {
        "procfs": is_linux_procfs(),
        "psi": read_pressure("cpu") is not None,
        "cgroup_v2": cgroup_v2,
        "proc_locks": (PROC / "locks").exists(),
        "kmsg_readable": os.access("/dev/kmsg", os.R_OK),
        "syslog_file": Path("/var/log/syslog").exists() or Path("/var/log/messages").exists(),
        "cgroup_v1": Path("/sys/fs/cgroup/cpu/cpu.shares").exists(),
        "smaps_rollup": (PROC / "self" / "smaps_rollup").exists(),
        "btf_ebpf": Path("/sys/kernel/btf/vmlinux").exists(),
        "bpftrace": bool(__import__("shutil").which("bpftrace")),
    }
