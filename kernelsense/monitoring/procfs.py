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


def clock_ticks() -> int:
    try:
        return os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return 100


def capabilities() -> dict[str, bool]:
    """Probe which kernel interfaces are available on this machine."""
    cgroup_v2 = Path("/sys/fs/cgroup/cgroup.controllers").exists()
    return {
        "procfs": is_linux_procfs(),
        "psi": read_pressure("cpu") is not None,
        "cgroup_v2": cgroup_v2,
        "proc_locks": (PROC / "locks").exists(),
        "kmsg_readable": os.access("/dev/kmsg", os.R_OK),
        "syslog_file": Path("/var/log/syslog").exists() or Path("/var/log/messages").exists(),
    }
