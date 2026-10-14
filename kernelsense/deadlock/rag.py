"""Resource Allocation Graph (RAG), wait-for graph and deadlock detection.

Edges:
    P -> R   request edge   (process waits for resource)
    R -> P   assignment edge (instance of resource held by process)

* Single-instance resources: a cycle in the RAG (equivalently in the wait-for
  graph) is necessary *and sufficient* for deadlock.
* Multi-instance resources: a cycle is necessary but not sufficient, so the
  detection algorithm (Banker's-style reduction using current requests) is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class DetectionResult:
    deadlocked: bool
    deadlocked_processes: list[str]
    cycles: list[list[str]]
    method: str
    finish_order: list[str] = field(default_factory=list)


class ResourceAllocationGraph:
    def __init__(self) -> None:
        self.processes: list[str] = []
        self.instances: dict[str, int] = {}
        self.held: dict[tuple[str, str], int] = {}  # (process, resource) -> count
        self.requested: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------- building
    def add_process(self, p: str) -> "ResourceAllocationGraph":
        if p not in self.processes:
            self.processes.append(p)
        return self

    def add_resource(self, r: str, instances: int = 1) -> "ResourceAllocationGraph":
        if instances <= 0:
            raise ValueError("instances must be positive")
        self.instances[r] = instances
        return self

    def assign(self, r: str, p: str, count: int = 1) -> "ResourceAllocationGraph":
        self._check(p, r)
        held_total = sum(c for (_, rr), c in self.held.items() if rr == r)
        if held_total + count > self.instances[r]:
            raise ValueError(f"{r} has only {self.instances[r]} instance(s)")
        self.held[(p, r)] = self.held.get((p, r), 0) + count
        return self

    def request(self, p: str, r: str, count: int = 1) -> "ResourceAllocationGraph":
        self._check(p, r)
        self.requested[(p, r)] = self.requested.get((p, r), 0) + count
        return self

    def _check(self, p: str, r: str) -> None:
        self.add_process(p)
        if r not in self.instances:
            raise ValueError(f"unknown resource {r}")

    @property
    def single_instance(self) -> bool:
        return all(n == 1 for n in self.instances.values())

    def available(self) -> dict[str, int]:
        return {r: n - sum(c for (_, rr), c in self.held.items() if rr == r) for r, n in self.instances.items()}

    # --------------------------------------------------------------- graphs
    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for p in self.processes:
            g.add_node(p, kind="process")
        for r, n in self.instances.items():
            g.add_node(r, kind="resource", instances=n)
        for (p, r), c in self.held.items():
            g.add_edge(r, p, kind="assignment", count=c)
        for (p, r), c in self.requested.items():
            g.add_edge(p, r, kind="request", count=c)
        return g

    def wait_for_graph(self) -> nx.DiGraph:
        """Collapse resources: Pi -> Pj if Pi requests a resource Pj holds."""
        g = nx.DiGraph()
        g.add_nodes_from(self.processes)
        for (pi, r), _ in self.requested.items():
            for (pj, rr), _ in self.held.items():
                if rr == r and pj != pi:
                    g.add_edge(pi, pj, resource=r)
        return g

    # ------------------------------------------------------------ detection
    def detect(self) -> DetectionResult:
        if self.single_instance:
            wfg = self.wait_for_graph()
            cycles = [c for c in nx.simple_cycles(wfg)]
            dead = sorted({p for c in cycles for p in c})
            return DetectionResult(bool(cycles), dead, cycles, "Cycle detection on wait-for graph (single instance)")
        return self._reduction()

    def _reduction(self) -> DetectionResult:
        """Deadlock detection algorithm for multiple instances (graph reduction)."""
        work = self.available()
        finish = {p: not any(c for (pp, _), c in self.held.items() if pp == p) and
                  not any(c for (pp, _), c in self.requested.items() if pp == p) for p in self.processes}
        order: list[str] = []
        progressed = True
        while progressed:
            progressed = False
            for p in self.processes:
                if finish[p]:
                    continue
                req = {r: self.requested.get((p, r), 0) for r in self.instances}
                if all(req[r] <= work[r] for r in self.instances):
                    for r in self.instances:
                        work[r] += self.held.get((p, r), 0)
                    finish[p] = True
                    order.append(p)
                    progressed = True
        dead = [p for p in self.processes if not finish[p]]
        cycles = [c for c in nx.simple_cycles(self.graph())]
        return DetectionResult(bool(dead), dead, cycles, "Graph reduction (multiple instances)", order)


# ------------------------------------------------------------------ presets
def dining_philosophers(n: int = 5) -> ResourceAllocationGraph:
    """Every philosopher holds the left fork and waits for the right: classic deadlock."""
    g = ResourceAllocationGraph()
    for i in range(n):
        g.add_resource(f"Fork{i}")
    for i in range(n):
        g.assign(f"Fork{i}", f"Phil{i}")
        g.request(f"Phil{i}", f"Fork{(i + 1) % n}")
    return g


def simple_deadlock() -> ResourceAllocationGraph:
    g = ResourceAllocationGraph()
    g.add_resource("R1").add_resource("R2")
    g.assign("R1", "P1").assign("R2", "P2")
    g.request("P1", "R2").request("P2", "R1")
    return g


def multi_instance_no_deadlock() -> ResourceAllocationGraph:
    """Has a cycle but no deadlock (Silberschatz fig. 8.3-style)."""
    g = ResourceAllocationGraph()
    g.add_resource("R1", 2).add_resource("R2", 2)
    g.assign("R1", "P2").assign("R1", "P3").assign("R2", "P1").assign("R2", "P4")
    g.request("P1", "R1").request("P3", "R2")
    return g
