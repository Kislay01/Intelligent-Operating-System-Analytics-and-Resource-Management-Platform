"""Scheduling recommendation engine.

Two stages:
1. **Workload characterisation** (rule-based): burst variability, convoy risk,
   priority spread, share of short jobs. These produce human-readable reasons.
2. **Simulation-validated ranking**: every candidate policy is simulated on the
   workload and scored against the chosen goal. The rules explain *why*, the
   simulation confirms *which*.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pandas as pd

from .algorithms import MLFQ, SJF, SRTF, FCFS, PriorityScheduler, RoundRobin, Scheduler
from .models import Job, ScheduleResult

GOALS = {
    "balanced": "Balance waiting time, response time and fairness",
    "interactive": "Minimise response time (desktop, servers)",
    "throughput": "Minimise turnaround / maximise throughput (batch jobs)",
    "fairness": "Treat every process equally, avoid starvation",
}

# weights over normalised metrics (lower is better for all)
GOAL_WEIGHTS = {
    "balanced": {"avg_waiting": 0.35, "avg_response": 0.25, "unfairness": 0.25, "context_switches": 0.15},
    "interactive": {"avg_response": 0.6, "avg_waiting": 0.15, "unfairness": 0.15, "context_switches": 0.10},
    "throughput": {"avg_turnaround": 0.6, "avg_waiting": 0.2, "context_switches": 0.2},
    "fairness": {"unfairness": 0.5, "max_waiting": 0.3, "avg_response": 0.2},
}


@dataclass
class WorkloadProfile:
    n_jobs: int
    mean_burst: float
    median_burst: float
    burst_cv: float
    max_min_ratio: float
    short_job_share: float
    priority_spread: int
    arrival_span: int
    convoy_risk: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Recommendation:
    goal: str
    best: str
    suggested_quantum: int
    profile: WorkloadProfile
    reasons: list[str]
    ranking: pd.DataFrame
    results: dict[str, ScheduleResult] = field(repr=False, default_factory=dict)
    linux_hint: str = ""


def suggest_quantum(jobs: list[Job]) -> int:
    """Rule of thumb: ~80% of CPU bursts should finish within one quantum."""
    bursts = sorted(j.burst for j in jobs)
    idx = max(0, int(0.8 * len(bursts)) - 1)
    return max(1, min(bursts[idx], 20))


def profile(jobs: list[Job], quantum: int | None = None) -> WorkloadProfile:
    bursts = [j.burst for j in jobs]
    mean = statistics.mean(bursts)
    sd = statistics.pstdev(bursts)
    q = quantum or suggest_quantum(jobs)
    first = min(jobs, key=lambda j: (j.arrival, j.pid))
    later_short = [j for j in jobs if j is not first and j.burst < first.burst / 3]
    convoy = first.burst >= 2 * mean and len(later_short) >= max(2, len(jobs) // 3)
    prios = [j.priority for j in jobs]
    return WorkloadProfile(
        n_jobs=len(jobs),
        mean_burst=round(mean, 2),
        median_burst=float(statistics.median(bursts)),
        burst_cv=round(sd / mean, 3) if mean else 0.0,
        max_min_ratio=round(max(bursts) / max(min(bursts), 1), 2),
        short_job_share=round(sum(b <= q for b in bursts) / len(bursts), 3),
        priority_spread=max(prios) - min(prios),
        arrival_span=max(j.arrival for j in jobs) - min(j.arrival for j in jobs),
        convoy_risk=convoy,
    )


def _reasons(p: WorkloadProfile, q: int, goal: str) -> list[str]:
    r = []
    if p.burst_cv > 0.75:
        r.append(f"Burst lengths vary a lot (CV = {p.burst_cv}). Shortest-job policies (SJF/SRTF) "
                 "cut average waiting time sharply on such workloads.")
    elif p.burst_cv < 0.3:
        r.append(f"Burst lengths are similar (CV = {p.burst_cv}), so FCFS and Round Robin behave almost "
                 "like SJF; the simpler policy is enough.")
    if p.convoy_risk:
        r.append("Convoy effect risk: a long job arrives first and many short jobs queue behind it. "
                 "FCFS will perform badly; preemption is recommended.")
    if p.priority_spread >= 10:
        r.append(f"Priorities differ widely (spread = {p.priority_spread}). Use priority scheduling with "
                 "aging so low-priority jobs cannot starve.")
    if goal == "interactive":
        r.append(f"For responsiveness, a time-sliced policy with quantum ~{q} (80% of bursts finish within "
                 "one slice) keeps response time low without excessive context switches.")
    if p.short_job_share >= 0.7:
        r.append(f"{int(p.short_job_share*100)}% of jobs fit in one quantum, so Round Robin/MLFQ add little "
                 "switching overhead here.")
    r.append("Note: SJF/SRTF need the next burst length, which the OS must predict "
             "(exponential averaging); results shown assume perfect knowledge.")
    return r


def _linux_hint(best: str) -> str:
    if best.startswith("SJF") or best.startswith("SRTF"):
        return ("Linux has no SJF class. Approximate it by giving short interactive tasks lower nice values "
                "and moving long batch jobs to SCHED_BATCH / higher nice.")
    if best.startswith("Round Robin"):
        return "Closest Linux equivalent: SCHED_RR for real-time tasks; CFS already time-slices normal tasks fairly."
    if best.startswith("Priority"):
        return "Map priorities to nice values (-20..19) or SCHED_FIFO/SCHED_RR real-time priorities."
    if best.startswith("MLFQ"):
        return "CFS/EEVDF achieve MLFQ-like behaviour: tasks that sleep often get favoured when they wake."
    return "FCFS resembles SCHED_FIFO among equal-priority real-time tasks."


def candidates(jobs: list[Job], quantum: int, practical_only: bool = False) -> list[Scheduler]:
    """Candidate policies. ``practical_only`` drops SJF/SRTF, which need future burst knowledge."""
    quanta = sorted({max(1, quantum // 2), quantum, quantum * 2})
    oracle = [] if practical_only else [SJF(), SRTF()]
    return [
        FCFS(), *oracle,
        *[RoundRobin(q) for q in quanta],
        PriorityScheduler(preemptive=False),
        PriorityScheduler(preemptive=True, aging_interval=max(5, quantum * 2)),
        MLFQ((quantum, quantum * 2), boost_period=max(30, quantum * 10)),
    ]


def recommend(
    jobs: list[Job], goal: str = "balanced", quantum: int | None = None, practical_only: bool = False
) -> Recommendation:
    if goal not in GOAL_WEIGHTS:
        raise ValueError(f"goal must be one of {list(GOAL_WEIGHTS)}")
    q = quantum or suggest_quantum(jobs)
    prof = profile(jobs, q)
    results = {s.name: s.run(jobs) for s in candidates(jobs, q, practical_only)}
    rows = []
    for name, res in results.items():
        s = res.summary()
        s["unfairness"] = round(1 - s["fairness"], 4)
        rows.append(s)
    df = pd.DataFrame(rows)

    weights = GOAL_WEIGHTS[goal]
    score = pd.Series(0.0, index=df.index)
    for metric, w in weights.items():
        col = df[metric].astype(float)
        rng = col.max() - col.min()
        score += w * ((col - col.min()) / rng if rng > 0 else 0.0)
    df["score"] = score.round(4)
    df = df.sort_values(["score", "avg_waiting"]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    best = df.iloc[0]["algorithm"]
    return Recommendation(
        goal=goal,
        best=best,
        suggested_quantum=q,
        profile=prof,
        reasons=_reasons(prof, q, goal),
        ranking=df,
        results=results,
        linux_hint=_linux_hint(best),
    )
