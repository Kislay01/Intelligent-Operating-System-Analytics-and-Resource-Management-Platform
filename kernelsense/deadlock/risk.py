"""Deadlock risk scoring.

Prediction here means *flagging states that are heading towards deadlock before
it happens*. The score combines the Coffman conditions that can be observed:

* unsafe state (Banker's safety check fails)      -> strongest signal
* resource scarcity (little of some resource free) -> requests will block
* hold-and-wait (processes holding while needing)  -> Coffman condition 2
* processes unable to proceed with current Available
* cycles / long waiting chains in the wait-for graph -> circular wait forming
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .bankers import BankersState
from .rag import ResourceAllocationGraph


@dataclass
class RiskReport:
    score: float  # 0..100
    level: str
    factors: dict[str, float]
    notes: list[str]


def _level(score: float) -> str:
    return "Critical" if score >= 80 else "High" if score >= 60 else "Medium" if score >= 30 else "Low"


def bankers_risk(state: BankersState) -> RiskReport:
    safety = state.safety()
    need = state.need
    total = state.total
    n = len(state.processes)

    scarcity = max(
        (1 - state.available[j] / total[j]) if total[j] else 0.0 for j in range(len(state.resources))
    )
    hold_and_wait = sum(
        1 for i in range(n) if any(state.allocation[i]) and any(need[i])
    ) / max(n, 1)
    can_run = sum(1 for i in range(n) if all(need[i][j] <= state.available[j] for j in range(len(total))))
    blocked_now = 1 - can_run / max(n, 1)

    score = 55 * (not safety.safe) + 20 * scarcity + 12.5 * hold_and_wait + 12.5 * blocked_now
    notes = []
    if not safety.safe:
        notes.append(f"Unsafe state: {', '.join(safety.blocked)} cannot be guaranteed to finish.")
    else:
        notes.append(f"Safe sequence exists: <{', '.join(safety.sequence)}>.")
    if scarcity > 0.8:
        notes.append("At least one resource type is more than 80% allocated.")
    if hold_and_wait > 0.5:
        notes.append("Most processes hold resources while waiting for more (hold-and-wait).")
    if can_run == 0:
        notes.append("No process can make progress with the currently available resources.")
    return RiskReport(
        round(score, 1),
        _level(score),
        {
            "unsafe": float(not safety.safe),
            "scarcity": round(scarcity, 3),
            "hold_and_wait": round(hold_and_wait, 3),
            "blocked_now": round(blocked_now, 3),
        },
        notes,
    )


def graph_risk(rag: ResourceAllocationGraph) -> RiskReport:
    det = rag.detect()
    wfg = rag.wait_for_graph()
    longest_chain = nx.dag_longest_path_length(wfg) if nx.is_directed_acyclic_graph(wfg) else len(rag.processes)
    waiting = sum(1 for p in rag.processes if any(pp == p for (pp, _) in rag.requested))
    holding_waiting = sum(
        1 for p in rag.processes
        if any(pp == p for (pp, _) in rag.requested) and any(pp == p for (pp, _) in rag.held)
    )
    n = max(len(rag.processes), 1)
    if det.deadlocked:
        score = 100.0
    else:
        score = 35 * bool(det.cycles) + 30 * (holding_waiting / n) + 20 * min(longest_chain / 4, 1) + 15 * (waiting / n)
    notes = []
    if det.deadlocked:
        notes.append(f"DEADLOCK: {', '.join(det.deadlocked_processes)} are waiting in a circular chain.")
    elif det.cycles:
        notes.append("A cycle exists, but enough instances are available to break it (no deadlock yet).")
    if longest_chain >= 2 and not det.deadlocked:
        notes.append(f"Waiting chain of length {longest_chain}: one more request could close a cycle.")
    if not notes:
        notes.append("No circular wait forming.")
    return RiskReport(
        round(min(score, 100.0), 1),
        _level(score),
        {
            "deadlocked": float(det.deadlocked),
            "cycles": float(len(det.cycles)),
            "longest_wait_chain": float(longest_chain),
            "hold_and_wait": round(holding_waiting / n, 3),
        },
        notes,
    )
