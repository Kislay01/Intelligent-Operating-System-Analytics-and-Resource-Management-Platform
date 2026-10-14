"""Rule-based workload classification from scheduler behaviour.

The key OS insight: a process that *voluntarily* gives up the CPU (blocking on
I/O, sleeping, waiting on a lock) accumulates voluntary context switches, while
a process that keeps running until the scheduler preempts it accumulates
involuntary (non-voluntary) context switches.

    CPU-bound  -> high CPU%, involuntary switches dominate
    I/O-bound  -> low CPU%, voluntary switches dominate, often disk activity
    Interactive-> moderate voluntary switching with small CPU bursts
    Idle       -> almost no CPU and almost no switching
"""

from __future__ import annotations

from dataclasses import dataclass

CPU_BOUND = "CPU-bound"
IO_BOUND = "I/O-bound"
INTERACTIVE = "Interactive"
IDLE = "Idle"
MIXED = "Mixed"

CLASSES = [CPU_BOUND, IO_BOUND, INTERACTIVE, IDLE, MIXED]


@dataclass
class WorkloadSignal:
    cpu_percent: float
    vol_ctx_delta: int
    invol_ctx_delta: int
    io_bytes_delta: int = 0
    interval_s: float = 1.0


def classify(sig: WorkloadSignal) -> str:
    interval = max(sig.interval_s, 1e-6)
    vol_rate = sig.vol_ctx_delta / interval
    invol_rate = sig.invol_ctx_delta / interval
    io_rate = sig.io_bytes_delta / interval
    total_switch_rate = vol_rate + invol_rate

    if sig.cpu_percent < 0.5 and total_switch_rate < 1 and io_rate < 1024:
        return IDLE
    if sig.cpu_percent >= 50 or (sig.cpu_percent >= 20 and invol_rate >= vol_rate):
        return CPU_BOUND
    if io_rate >= 64 * 1024 or (vol_rate > 5 * max(invol_rate, 0.1) and vol_rate >= 20):
        return IO_BOUND
    if vol_rate >= invol_rate and sig.cpu_percent < 20:
        return INTERACTIVE
    return MIXED


def voluntary_ratio(vol: int, invol: int) -> float:
    """Fraction of switches that were voluntary (1.0 = purely I/O/blocking)."""
    total = vol + invol
    return vol / total if total else 0.0
