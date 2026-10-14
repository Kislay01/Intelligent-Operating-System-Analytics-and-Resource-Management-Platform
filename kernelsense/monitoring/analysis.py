"""Process-level analysis: process tree, anomaly detection and summaries."""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from .collector import ProcessInfo
from .procfs import STATE_NAMES

# Daemons that legitimately run with init (PID 1) as parent.
KNOWN_INIT_CHILDREN = {
    "systemd", "systemd-journald", "systemd-logind", "systemd-udevd", "systemd-resolved",
    "systemd-networkd", "systemd-timesyncd", "dbus-daemon", "cron", "crond", "sshd", "rsyslogd",
    "init", "login", "agetty", "containerd", "dockerd", "snapd", "polkitd", "NetworkManager",
    "udisksd", "accounts-daemon", "wsl-pro-service", "unattended-upgr",
}


@dataclass
class Anomaly:
    pid: int
    name: str
    kind: str
    severity: str  # info | warning | critical
    detail: str


def to_frame(procs: list[ProcessInfo]) -> pd.DataFrame:
    if not procs:
        return pd.DataFrame()
    df = pd.DataFrame([p.to_dict() for p in procs])
    df["state_name"] = df["state"].map(STATE_NAMES).fillna("Unknown")
    df["age_s"] = (time.time() - df["create_time"]).clip(lower=0).round(0)
    return df


def state_distribution(procs: list[ProcessInfo]) -> dict[str, int]:
    counts = Counter(STATE_NAMES.get(p.state, p.state) for p in procs)
    return dict(counts.most_common())


def build_tree(procs: list[ProcessInfo]) -> dict[int, list[int]]:
    """Parent PID -> children PIDs."""
    children: dict[int, list[int]] = defaultdict(list)
    pids = {p.pid for p in procs}
    for p in procs:
        parent = p.ppid if p.ppid in pids else 0
        if p.pid != parent:
            children[parent].append(p.pid)
    for v in children.values():
        v.sort()
    return dict(children)


def render_tree(procs: list[ProcessInfo], root: int = 0, max_depth: int = 6, max_lines: int = 400) -> str:
    """Text rendering similar to `pstree -p`."""
    by_pid = {p.pid: p for p in procs}
    tree = build_tree(procs)
    lines: list[str] = []

    def walk(pid: int, prefix: str, depth: int) -> None:
        kids = tree.get(pid, [])
        for i, child in enumerate(kids):
            if len(lines) >= max_lines:
                return
            last = i == len(kids) - 1
            p = by_pid[child]
            lines.append(f"{prefix}{'└── ' if last else '├── '}{p.name} ({p.pid}) [{p.state}]")
            if depth < max_depth:
                walk(child, prefix + ("    " if last else "│   "), depth + 1)

    walk(root, "", 0)
    return "\n".join(lines)


def tree_edges(procs: list[ProcessInfo]) -> pd.DataFrame:
    """Edges for a sunburst/treemap view (ids, parents, labels, values)."""
    pids = {p.pid for p in procs}
    rows = []
    for p in procs:
        parent = str(p.ppid) if p.ppid in pids and p.ppid != p.pid else ""
        rows.append({"id": str(p.pid), "parent": parent, "label": f"{p.name} ({p.pid})",
                     "rss_mb": max(p.rss_mb, 0.01), "cpu": p.cpu_percent})
    return pd.DataFrame(rows)


def detect_anomalies(
    procs: list[ProcessInfo],
    child_threshold: int = 50,
    thread_threshold: int = 500,
    fd_threshold: int = 1000,
    cpu_threshold: float = 90.0,
) -> list[Anomaly]:
    out: list[Anomaly] = []
    children_count = Counter(p.ppid for p in procs)
    me = os.getuid() if hasattr(os, "getuid") else None
    by_pid = {p.pid: p for p in procs}

    for p in procs:
        if p.state == "Z":
            parent = by_pid.get(p.ppid)
            out.append(Anomaly(p.pid, p.name, "Zombie process", "warning",
                               f"Exited but not reaped by parent {parent.name if parent else '?'} ({p.ppid})."))
        if p.state == "D":
            out.append(Anomaly(p.pid, p.name, "Uninterruptible sleep", "warning",
                               "Blocked in the kernel (usually I/O). Long D-state can indicate a hung task."))
        if p.ppid == 1 and p.name not in KNOWN_INIT_CHILDREN and p.username not in ("root", "?"):
            out.append(Anomaly(p.pid, p.name, "Orphan process", "info",
                               "Re-parented to init (PID 1); its original parent exited."))
        # kthreadd (PID 2) is the parent of every kernel thread by design.
        if p.pid not in (0, 1, 2) and children_count.get(p.pid, 0) >= child_threshold:
            out.append(Anomaly(p.pid, p.name, "Possible fork storm", "critical",
                               f"{children_count[p.pid]} direct children."))
        if p.num_threads >= thread_threshold:
            out.append(Anomaly(p.pid, p.name, "Thread explosion", "warning", f"{p.num_threads} threads."))
        if p.num_fds >= fd_threshold:
            out.append(Anomaly(p.pid, p.name, "Possible FD leak", "warning", f"{p.num_fds} open file descriptors."))
        if p.cpu_percent >= cpu_threshold:
            out.append(Anomaly(p.pid, p.name, "CPU hog", "info", f"{p.cpu_percent:.0f}% CPU."))
        if p.majflt > 10000 and me is not None:
            out.append(Anomaly(p.pid, p.name, "Heavy major faults", "info",
                               f"{p.majflt} major page faults (pages loaded from disk)."))
    order = {"critical": 0, "warning": 1, "info": 2}
    out.sort(key=lambda a: order[a.severity])
    return out
