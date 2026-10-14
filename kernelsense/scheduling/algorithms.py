"""CPU scheduling algorithms implemented as discrete-time simulators.

Every scheduler implements ``Scheduler.run(jobs) -> ScheduleResult`` so new
policies can be added without touching the recommender or the UI.

Implemented:
    FCFS, SJF (non-preemptive), SRTF (preemptive SJF), Round Robin,
    Priority (preemptive / non-preemptive, optional aging), MLFQ.

Conventions:
    * time is an integer tick
    * ties are broken by arrival time, then PID (deterministic output)
    * a context switch is counted every time the CPU moves from one process
      to a *different* process (idle gaps are not counted)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from .models import IDLE, Job, ProcessStats, ScheduleResult, Segment


class _Timeline:
    """Records execution one slice at a time and merges adjacent slices."""

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.switches = 0
        self._last_pid: str | None = None

    def run(self, pid: str, start: int, end: int, queue: int | None = None) -> None:
        if end <= start:
            return
        if pid != IDLE:
            if self._last_pid is not None and self._last_pid != pid:
                self.switches += 1
            self._last_pid = pid
        last = self.segments[-1] if self.segments else None
        if last and last.pid == pid and last.end == start and last.queue == queue:
            last.end = end
        else:
            self.segments.append(Segment(pid, start, end, queue))


def _validate(jobs: list[Job]) -> list[Job]:
    if not jobs:
        raise ValueError("At least one job is required")
    pids = [j.pid for j in jobs]
    if len(set(pids)) != len(pids):
        raise ValueError("Job PIDs must be unique")
    return sorted(jobs, key=lambda j: (j.arrival, j.pid))


def _build_stats(jobs: list[Job], first_run: dict[str, int], completion: dict[str, int]) -> list[ProcessStats]:
    return [
        ProcessStats(j.pid, j.arrival, j.burst, j.priority, first_run[j.pid], completion[j.pid]) for j in jobs
    ]


class Scheduler(ABC):
    name: str = "Scheduler"
    preemptive: bool = False

    @abstractmethod
    def run(self, jobs: list[Job]) -> ScheduleResult: ...

    def params(self) -> dict:
        return {}


# --------------------------------------------------------------------------- #
# Non-preemptive policies: pick a job, run it to completion.
# --------------------------------------------------------------------------- #
class _NonPreemptive(Scheduler):
    def _key(self, job: Job, t: int, waited: int):  # lower is better
        raise NotImplementedError

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs = _validate(jobs)
        pending = list(jobs)
        ready: list[Job] = []
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        while pending or ready:
            while pending and pending[0].arrival <= t:
                ready.append(pending.pop(0))
            if not ready:
                nxt = pending[0].arrival
                tl.run(IDLE, t, nxt)
                t = nxt
                continue
            job = min(ready, key=lambda j: self._key(j, t, t - j.arrival))
            ready.remove(job)
            first[job.pid] = t
            tl.run(job.pid, t, t + job.burst)
            t += job.burst
            done[job.pid] = t
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


class FCFS(_NonPreemptive):
    name = "FCFS"

    def _key(self, job, t, waited):
        return (job.arrival, job.pid)


class SJF(_NonPreemptive):
    name = "SJF"

    def _key(self, job, t, waited):
        return (job.burst, job.arrival, job.pid)


# --------------------------------------------------------------------------- #
# Tick-based preemptive engine shared by SRTF and preemptive Priority.
# --------------------------------------------------------------------------- #
class _TickPreemptive(Scheduler):
    preemptive = True

    def _key(self, job: Job, remaining: int, waited: int):
        raise NotImplementedError

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs = _validate(jobs)
        remaining = {j.pid: j.burst for j in jobs}
        last_ready = {j.pid: j.arrival for j in jobs}  # for aging: time it last started waiting
        by_pid = {j.pid: j for j in jobs}
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        current: str | None = None
        while len(done) < len(jobs):
            ready = [p for p, r in remaining.items() if r > 0 and by_pid[p].arrival <= t]
            if not ready:
                nxt = min(by_pid[p].arrival for p, r in remaining.items() if r > 0)
                tl.run(IDLE, t, nxt)
                t = nxt
                current = None
                continue
            pick = min(ready, key=lambda p: self._key(by_pid[p], remaining[p], t - last_ready[p]))
            if current is not None and current != pick and remaining.get(current, 0) > 0:
                last_ready[current] = t
            current = pick
            first.setdefault(pick, t)
            tl.run(pick, t, t + 1)
            remaining[pick] -= 1
            t += 1
            if remaining[pick] == 0:
                done[pick] = t
                current = None
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


class SRTF(_TickPreemptive):
    name = "SRTF"

    def _key(self, job, remaining, waited):
        return (remaining, job.arrival, job.pid)


class PriorityScheduler(Scheduler):
    """Priority scheduling (lower number = higher priority).

    ``aging``: every ``aging_interval`` ticks spent waiting improves the effective
    priority by 1, which prevents starvation of low-priority jobs.
    """

    def __init__(self, preemptive: bool = True, aging_interval: int | None = None) -> None:
        self.preemptive = preemptive
        self.aging_interval = aging_interval
        self.name = f"Priority ({'P' if preemptive else 'NP'}{', aging' if aging_interval else ''})"

    def params(self) -> dict:
        return {"preemptive": self.preemptive, "aging_interval": self.aging_interval}

    def _effective(self, job: Job, waited: int) -> float:
        if self.aging_interval:
            return job.priority - waited // self.aging_interval
        return job.priority

    def run(self, jobs: list[Job]) -> ScheduleResult:
        sched = self

        if self.preemptive:
            class _P(_TickPreemptive):
                name = sched.name

                def _key(self, job, remaining, waited):
                    return (sched._effective(job, waited), job.arrival, job.pid)

                def params(self):
                    return sched.params()

            return _P().run(jobs)

        class _NP(_NonPreemptive):
            name = sched.name

            def _key(self, job, t, waited):
                return (sched._effective(job, waited), job.arrival, job.pid)

            def params(self):
                return sched.params()

        return _NP().run(jobs)


class RoundRobin(Scheduler):
    preemptive = True

    def __init__(self, quantum: int = 4) -> None:
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        self.quantum = quantum
        self.name = f"Round Robin (q={quantum})"

    def params(self) -> dict:
        return {"quantum": self.quantum}

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs = _validate(jobs)
        pending = deque(jobs)
        queue: deque[Job] = deque()
        remaining = {j.pid: j.burst for j in jobs}
        tl = _Timeline()
        first, done = {}, {}
        t = 0

        def admit(upto: int) -> None:
            while pending and pending[0].arrival <= upto:
                queue.append(pending.popleft())

        admit(t)
        while queue or pending:
            if not queue:
                nxt = pending[0].arrival
                tl.run(IDLE, t, nxt)
                t = nxt
                admit(t)
                continue
            job = queue.popleft()
            first.setdefault(job.pid, t)
            run_for = min(self.quantum, remaining[job.pid])
            tl.run(job.pid, t, t + run_for)
            t += run_for
            remaining[job.pid] -= run_for
            # Newly arrived jobs enter the queue before the preempted job (textbook convention).
            admit(t)
            if remaining[job.pid] > 0:
                queue.append(job)
            else:
                done[job.pid] = t
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


class MLFQ(Scheduler):
    """Multi-Level Feedback Queue.

    Rules (Arpaci-Dusseau, OSTEP ch. 8):
      1. Higher queue runs first; round robin within a queue.
      2. New jobs enter the top queue.
      3. A job that uses its whole quantum is demoted one level.
      4. A job preempted by a higher-queue arrival keeps its level.
      5. Every ``boost_period`` ticks all jobs move back to the top queue
         (prevents starvation and lets phase-changing jobs be re-classified).
    The last queue uses FCFS (quantum = infinity).
    """

    preemptive = True

    def __init__(self, quanta: tuple[int, ...] = (4, 8), boost_period: int | None = 50) -> None:
        self.quanta = tuple(quanta)
        self.boost_period = boost_period
        self.levels = len(self.quanta) + 1
        self.name = f"MLFQ ({'/'.join(map(str, self.quanta))}/FCFS)"

    def params(self) -> dict:
        return {"quanta": list(self.quanta), "boost_period": self.boost_period}

    def _quantum(self, level: int) -> float:
        return self.quanta[level] if level < len(self.quanta) else float("inf")

    def run(self, jobs: list[Job]) -> ScheduleResult:
        jobs = _validate(jobs)
        pending = deque(jobs)
        queues: list[deque[str]] = [deque() for _ in range(self.levels)]
        level = {j.pid: 0 for j in jobs}
        remaining = {j.pid: j.burst for j in jobs}
        used = {j.pid: 0 for j in jobs}  # ticks used in current quantum
        tl = _Timeline()
        first, done = {}, {}
        t = 0
        last_boost = 0
        current: str | None = None

        def admit() -> None:
            while pending and pending[0].arrival <= t:
                queues[0].append(pending.popleft().pid)

        while len(done) < len(jobs):
            admit()
            if self.boost_period and t - last_boost >= self.boost_period:
                last_boost = t
                everyone = [p for q in queues for p in q]
                for q in queues:
                    q.clear()
                if current is not None:
                    everyone.insert(0, current)
                    current = None
                for p in everyone:
                    level[p] = 0
                    used[p] = 0
                    queues[0].append(p)

            top = next((i for i, q in enumerate(queues) if q), None)
            if current is not None and top is not None and top < level[current]:
                # preempted by a higher-priority queue: keep level, keep remaining quantum
                queues[level[current]].appendleft(current)
                current = None
            if current is None:
                top = next((i for i, q in enumerate(queues) if q), None)
                if top is None:
                    nxt = pending[0].arrival
                    tl.run(IDLE, t, nxt)
                    t = nxt
                    continue
                current = queues[top].popleft()

            pid = current
            first.setdefault(pid, t)
            tl.run(pid, t, t + 1, queue=level[pid])
            t += 1
            remaining[pid] -= 1
            used[pid] += 1
            if remaining[pid] == 0:
                done[pid] = t
                current = None
            elif used[pid] >= self._quantum(level[pid]):
                used[pid] = 0
                level[pid] = min(level[pid] + 1, self.levels - 1)
                admit()
                queues[level[pid]].append(pid)
                current = None
        return ScheduleResult(self.name, tl.segments, _build_stats(jobs, first, done), tl.switches, self.params())


def default_schedulers(quantum: int = 4) -> list[Scheduler]:
    return [
        FCFS(),
        SJF(),
        SRTF(),
        RoundRobin(quantum),
        PriorityScheduler(preemptive=False),
        PriorityScheduler(preemptive=True, aging_interval=10),
        MLFQ((quantum, quantum * 2)),
    ]


def compare(jobs: list[Job], schedulers: list[Scheduler] | None = None) -> list[ScheduleResult]:
    return [s.run(jobs) for s in (schedulers or default_schedulers())]
