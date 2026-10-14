"""Proportional-share schedulers modelled on real kernels.

* CFS    - Linux Completely Fair Scheduler (2.6.23 - 6.5): run the task with
           the smallest *virtual runtime*; vruntime grows inversely to the
           task's weight (from its nice value). Time slices come from
           sched_latency split by weight, bounded below by min_granularity.
* EEVDF  - Linux >= 6.6: among *eligible* tasks (vruntime <= weighted average
           vruntime, i.e. lag >= 0) pick the one with the earliest *virtual
           deadline* = vruntime + slice / weight.
* Stride - deterministic proportional share (Waldspurger): each task has a
           stride = BIG / tickets; run the minimum pass, then pass += stride.
* Lottery- randomized proportional share: draw a ticket, run its owner for a
           quantum.

``Job.priority`` is interpreted as a relative nice level: the smallest priority
in the workload maps to nice 0 and every step above it to nice +1 (capped at
19), so weights follow the kernel's nice-to-weight table.
"""

from __future__ import annotations

import random

from .algorithms import Scheduler, _Timeline, _build_stats, _validate
from .models import IDLE, Job, ScheduleResult

# Linux kernel sched_prio_to_weight[] for nice -20 .. 19.
NICE_TO_WEIGHT = [
    88761, 71755, 56483, 46273, 36291,
    29154, 23254, 18705, 14949, 11916,
    9548, 7620, 6100, 4904, 3906,
    3121, 2501, 1991, 1586, 1277,
    1024, 820, 655, 526, 423,
    335, 272, 215, 172, 137,
    110, 87, 70, 56, 45,
    36, 29, 23, 18, 15,
]
NICE_0_WEIGHT = 1024


def nice_weight(nice: int) -> int:
    return NICE_TO_WEIGHT[max(-20, min(19, nice)) + 20]


def relative_nice(jobs: list[Job]) -> dict[str, int]:
    base = min(j.priority for j in jobs)
    return {j.pid: min(19, j.priority - base) for j in jobs}


class _ProportionalBase(Scheduler):
    preemptive = True

    def _setup(self, jobs: list[Job]):
        jobs = _validate(jobs)
        nice = relative_nice(jobs)
        weight = {j.pid: nice_weight(nice[j.pid]) for j in jobs}
        return jobs, nice, weight


class CFS(_ProportionalBase):
    """Tick-based CFS model.

    sched_latency: target period in which every runnable task runs once.
    min_granularity: minimum slice, so many tasks do not cause excessive switching.
    New tasks start at the current min_vruntime (so they neither starve others
    nor get starved).
    """

    def __init__(self, sched_latency: int = 6, min_granularity: int = 1) -> None:
        self.sched_latency = sched_latency
        self.min_granularity = min_granularity
        self.name = f"CFS (lat={sched_latency})"

    def params(self) -> dict:
        return {"sched_latency": self.sched_latency, "min_granularity": self.min_granularity}

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs, _, weight = self._setup(jobs)
        pending = list(jobs)
        remaining = {j.pid: j.burst for j in jobs}
        vruntime: dict[str, float] = {}
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        current: str | None = None
        ran_in_slice = 0
        min_vr = 0.0
        while len(done) < len(jobs):
            while pending and pending[0].arrival <= t:
                j = pending.pop(0)
                vruntime[j.pid] = min_vr  # place new task at min_vruntime
            runnable = [p for p in vruntime if remaining[p] > 0]
            if not runnable:
                nxt = pending[0].arrival
                tl.run(IDLE, t, nxt)
                t, current, ran_in_slice = nxt, None, 0
                continue
            total_w = sum(weight[p] for p in runnable)
            if current is not None and remaining[current] > 0:
                slice_len = max(self.min_granularity, self.sched_latency * weight[current] / total_w)
                leftmost = min(runnable, key=lambda p: (vruntime[p], p))
                if ran_in_slice >= slice_len and leftmost != current and vruntime[leftmost] < vruntime[current]:
                    current, ran_in_slice = leftmost, 0
            else:
                current = min(runnable, key=lambda p: (vruntime[p], p))
                ran_in_slice = 0
            first.setdefault(current, t)
            tl.run(current, t, t + 1)
            t += 1
            ran_in_slice += 1
            remaining[current] -= 1
            vruntime[current] += NICE_0_WEIGHT / weight[current]
            live = [vruntime[p] for p in vruntime if remaining[p] > 0]
            if live:
                min_vr = max(min_vr, min(live))
            if remaining[current] == 0:
                done[current] = t
                current, ran_in_slice = None, 0
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


class EEVDF(_ProportionalBase):
    """Earliest Eligible Virtual Deadline First (simplified, tick-based).

    Each task requests a ``base_slice``. Virtual time advances by 1/weight per
    unit of service. A task is eligible when its vruntime is not ahead of the
    weighted average (non-negative lag); among eligible tasks the earliest
    virtual deadline (vruntime + base_slice / weight) runs. A running task is
    re-evaluated once it has consumed its slice.
    """

    def __init__(self, base_slice: int = 3) -> None:
        self.base_slice = base_slice
        self.name = f"EEVDF (slice={base_slice})"

    def params(self) -> dict:
        return {"base_slice": self.base_slice}

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs, _, weight = self._setup(jobs)
        pending = list(jobs)
        remaining = {j.pid: j.burst for j in jobs}
        vr: dict[str, float] = {}
        deadline: dict[str, float] = {}
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        current: str | None = None
        used = 0

        def avg_vruntime(runnable):
            tw = sum(weight[p] for p in runnable)
            return sum(vr[p] * weight[p] for p in runnable) / tw

        while len(done) < len(jobs):
            while pending and pending[0].arrival <= t:
                j = pending.pop(0)
                live = [p for p in vr if remaining[p] > 0]
                vr[j.pid] = avg_vruntime(live) if live else 0.0  # join with zero lag
                deadline[j.pid] = vr[j.pid] + self.base_slice * NICE_0_WEIGHT / weight[j.pid]
            runnable = [p for p in vr if remaining[p] > 0]
            if not runnable:
                nxt = pending[0].arrival
                tl.run(IDLE, t, nxt)
                t, current, used = nxt, None, 0
                continue
            if current is None or remaining[current] == 0 or used >= self.base_slice:
                if current is not None and remaining[current] > 0:
                    deadline[current] = vr[current] + self.base_slice * NICE_0_WEIGHT / weight[current]
                avg = avg_vruntime(runnable)
                eligible = [p for p in runnable if vr[p] <= avg + 1e-9] or runnable
                current = min(eligible, key=lambda p: (deadline[p], p))
                used = 0
            first.setdefault(current, t)
            tl.run(current, t, t + 1)
            t += 1
            used += 1
            remaining[current] -= 1
            vr[current] += NICE_0_WEIGHT / weight[current]
            if remaining[current] == 0:
                done[current] = t
                current, used = None, 0
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


class Stride(_ProportionalBase):
    BIG = 1 << 20

    def __init__(self, quantum: int = 1) -> None:
        self.quantum = quantum
        self.name = f"Stride (q={quantum})"

    def params(self) -> dict:
        return {"quantum": self.quantum}

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs, _, tickets = self._setup(jobs)
        pending = list(jobs)
        remaining = {j.pid: j.burst for j in jobs}
        stride = {p: self.BIG / w for p, w in tickets.items()}
        passv: dict[str, float] = {}
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        while len(done) < len(jobs):
            while pending and pending[0].arrival <= t:
                j = pending.pop(0)
                live = [passv[p] for p in passv if remaining[p] > 0]
                passv[j.pid] = min(live) if live else 0.0
            runnable = [p for p in passv if remaining[p] > 0]
            if not runnable:
                nxt = pending[0].arrival
                tl.run(IDLE, t, nxt)
                t = nxt
                continue
            p = min(runnable, key=lambda x: (passv[x], x))
            run_for = min(self.quantum, remaining[p])
            first.setdefault(p, t)
            tl.run(p, t, t + run_for)
            t += run_for
            remaining[p] -= run_for
            passv[p] += stride[p] * run_for
            if remaining[p] == 0:
                done[p] = t
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


class Lottery(_ProportionalBase):
    def __init__(self, quantum: int = 1, seed: int = 0) -> None:
        self.quantum = quantum
        self.seed = seed
        self.name = f"Lottery (q={quantum})"

    def params(self) -> dict:
        return {"quantum": self.quantum, "seed": self.seed}

    def run(self, jobs: list[Job]) -> ScheduleResult:
        rng = random.Random(self.seed)
        jobs, _, tickets = self._setup(jobs)
        pending = list(jobs)
        remaining = {j.pid: j.burst for j in jobs}
        active: list[str] = []
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        while len(done) < len(jobs):
            while pending and pending[0].arrival <= t:
                active.append(pending.pop(0).pid)
            runnable = [p for p in active if remaining[p] > 0]
            if not runnable:
                nxt = pending[0].arrival
                tl.run(IDLE, t, nxt)
                t = nxt
                continue
            draw = rng.uniform(0, sum(tickets[p] for p in runnable))
            acc = 0.0
            winner = runnable[-1]
            for p in runnable:
                acc += tickets[p]
                if draw <= acc:
                    winner = p
                    break
            run_for = min(self.quantum, remaining[winner])
            first.setdefault(winner, t)
            tl.run(winner, t, t + run_for)
            t += run_for
            remaining[winner] -= run_for
            if remaining[winner] == 0:
                done[winner] = t
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


def cpu_share(result: ScheduleResult, start: int, end: int) -> dict[str, float]:
    """Fraction of CPU each process received in [start, end) - used to check proportional share."""
    got: dict[str, int] = {}
    for seg in result.gantt:
        if seg.pid == IDLE:
            continue
        overlap = max(0, min(seg.end, end) - max(seg.start, start))
        if overlap:
            got[seg.pid] = got.get(seg.pid, 0) + overlap
    total = sum(got.values()) or 1
    return {p: v / total for p, v in got.items()}
