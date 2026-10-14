"""Live deadlock indicators from the running Linux system.

* /proc/locks lists POSIX/flock file locks. Lines containing ``->`` are
  *blocked waiters* on the lock described by the preceding line, which gives us
  real holder -> waiter edges between PIDs to build a wait-for graph.
* Processes stuck in uninterruptible sleep (state D) are reported too, since a
  task stuck in D for long is how the kernel's hung-task detector spots kernel
  level deadlocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..monitoring.procfs import read_stat


@dataclass
class FileLock:
    lock_id: str
    kind: str  # POSIX / FLOCK / OFDLCK / LEASE
    mode: str  # ADVISORY / MANDATORY
    access: str  # READ / WRITE
    pid: int
    inode: str
    blocked: bool


def parse_proc_locks(text: str) -> list[FileLock]:
    locks: list[FileLock] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        lock_id, _, rest = line.partition(":")
        parts = rest.split()
        blocked = False
        if parts and parts[0] == "->":
            blocked = True
            parts = parts[1:]
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[3])
        except ValueError:
            continue
        locks.append(FileLock(lock_id.strip(), parts[0], parts[1], parts[2], pid, parts[4], blocked))
    return locks


def read_file_locks(path: str = "/proc/locks") -> list[FileLock]:
    try:
        return parse_proc_locks(Path(path).read_text())
    except OSError:
        return []


def file_lock_wait_for(locks: list[FileLock]) -> nx.DiGraph:
    """Waiter -> holder edges (same lock id)."""
    g = nx.DiGraph()
    holders = {l.lock_id: l.pid for l in locks if not l.blocked}
    for l in locks:
        g.add_node(l.pid)
        if l.blocked and l.lock_id in holders and holders[l.lock_id] != l.pid:
            g.add_edge(l.pid, holders[l.lock_id], inode=l.inode)
    return g


def live_report(proc_locks_path: str = "/proc/locks") -> dict:
    locks = read_file_locks(proc_locks_path)
    wfg = file_lock_wait_for(locks)
    cycles = list(nx.simple_cycles(wfg))
    d_state = []
    for entry in Path("/proc").iterdir() if Path("/proc").exists() else []:
        if entry.name.isdigit():
            st = read_stat(int(entry.name))
            if st and st["state"] == "D":
                d_state.append({"pid": int(entry.name), "name": st["comm"]})
    return {
        "locks": locks,
        "held": sum(not l.blocked for l in locks),
        "waiting": sum(l.blocked for l in locks),
        "wait_for_graph": wfg,
        "cycles": cycles,
        "deadlock": bool(cycles),
        "d_state": d_state,
    }
