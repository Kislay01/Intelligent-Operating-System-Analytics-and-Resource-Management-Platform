"""OOM-killer victim prediction.

When memory and swap are exhausted, Linux's OOM killer picks the task with the
highest *badness*:

    badness = rss + swap + page-table pages  +  oom_score_adj * totalpages / 1000

/proc/[pid]/oom_score reports it normalised (Linux >= 5.9, fs/proc/base.c):

    oom_score = (1000 + badness * 1000 / totalpages) * 2 / 3

where totalpages = RAM + swap. 
This module ranks live processes by the kernel's own score, recomputes the
estimate from /proc data to show *why* each process ranks where it does, and
combines it with the system memory forecast to say which process the kernel
would kill and roughly when.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from ..monitoring import procfs


@dataclass
class OOMCandidate:
    pid: int
    name: str
    oom_score: int
    oom_score_adj: int
    estimated_score: int
    rss_mb: float
    swap_mb: float


def _status_kb(status: dict, key: str) -> int:
    try:
        return int(status.get(key, "0 kB").split()[0])
    except (ValueError, IndexError):
        return 0


def candidates(top: int = 10) -> list[OOMCandidate]:
    if not procfs.is_linux_procfs():
        return []
    mem = procfs.read_meminfo()
    total_kb = mem.get("MemTotal", 1) + mem.get("SwapTotal", 0)
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        score = procfs.read_oom_score(pid)
        status = procfs.read_status(pid)
        if not score or not status:
            continue
        rss = _status_kb(status, "VmRSS")
        swap = _status_kb(status, "VmSwap")
        pte = _status_kb(status, "VmPTE")
        badness = (rss + swap + pte) + score[1] * total_kb / 1000  # in kB, same unit as total_kb
        est = int((1000 + badness * 1000 / total_kb) * 2 / 3)
        if pid == 1 or score[1] == -1000:  # init and OOM-disabled tasks are never selected
            est = 0
        out.append(OOMCandidate(pid, status.get("Name", "?"), score[0], score[1], max(est, 0),
                                round(rss / 1024, 1), round(swap / 1024, 1)))
    out.sort(key=lambda c: c.oom_score, reverse=True)
    return out[:top]


def frame(top: int = 10) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in candidates(top)])


def prediction(eta_seconds: float | None, top: int = 5) -> dict:
    cands = candidates(top)
    victim = cands[0] if cands else None
    if victim is None:
        return {"victim": None, "message": "OOM scores are not available on this system."}
    if eta_seconds is None:
        msg = (f"No memory exhaustion forecast. If it happened now, the OOM killer would choose "
               f"{victim.name} (PID {victim.pid}, oom_score {victim.oom_score}).")
    else:
        msg = (f"Memory is forecast to run out in ~{eta_seconds / 60:.1f} min; the OOM killer would then kill "
               f"{victim.name} (PID {victim.pid}, oom_score {victim.oom_score}, RSS {victim.rss_mb} MB).")
    return {"victim": victim, "candidates": cands, "message": msg}
