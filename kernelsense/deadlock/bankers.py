"""Banker's algorithm (Dijkstra) for deadlock avoidance.

State:
    Available[m]       free instances of each resource type
    Max[n][m]          maximum demand of each process
    Allocation[n][m]   currently held instances
    Need = Max - Allocation

Safety algorithm: the state is *safe* if there is an order (safe sequence) in
which every process can obtain its remaining Need, finish, and release what it
holds. Resource-request algorithm: tentatively grant a request and keep it only
if the resulting state is still safe.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class SafetyStep:
    process: str
    need: list[int]
    work_before: list[int]
    work_after: list[int]


@dataclass
class SafetyResult:
    safe: bool
    sequence: list[str]
    steps: list[SafetyStep]
    blocked: list[str] = field(default_factory=list)  # processes that could never finish


@dataclass
class RequestResult:
    granted: bool
    reason: str
    safety: SafetyResult | None = None
    new_state: "BankersState | None" = None


@dataclass
class BankersState:
    processes: list[str]
    resources: list[str]
    available: list[int]
    maximum: list[list[int]]
    allocation: list[list[int]]

    def __post_init__(self) -> None:
        n, m = len(self.processes), len(self.resources)
        if len(self.available) != m:
            raise ValueError("available must have one entry per resource")
        for name, mat in (("maximum", self.maximum), ("allocation", self.allocation)):
            if len(mat) != n or any(len(row) != m for row in mat):
                raise ValueError(f"{name} must be {n}x{m}")
        for i in range(n):
            for j in range(m):
                if self.allocation[i][j] > self.maximum[i][j]:
                    raise ValueError(f"{self.processes[i]} holds more {self.resources[j]} than its maximum")
                if self.allocation[i][j] < 0 or self.maximum[i][j] < 0:
                    raise ValueError("matrix entries must be non-negative")

    @property
    def need(self) -> list[list[int]]:
        return [[mx - al for mx, al in zip(mrow, arow)] for mrow, arow in zip(self.maximum, self.allocation)]

    @property
    def total(self) -> list[int]:
        return [self.available[j] + sum(row[j] for row in self.allocation) for j in range(len(self.resources))]

    def copy(self) -> "BankersState":
        return deepcopy(self)

    # ---------------------------------------------------------------- safety
    def safety(self) -> SafetyResult:
        work = list(self.available)
        finish = [False] * len(self.processes)
        need = self.need
        sequence, steps = [], []
        progressed = True
        while progressed:
            progressed = False
            for i, name in enumerate(self.processes):
                if not finish[i] and all(need[i][j] <= work[j] for j in range(len(work))):
                    before = list(work)
                    work = [w + a for w, a in zip(work, self.allocation[i])]
                    finish[i] = True
                    sequence.append(name)
                    steps.append(SafetyStep(name, list(need[i]), before, list(work)))
                    progressed = True
        blocked = [p for p, f in zip(self.processes, finish) if not f]
        return SafetyResult(all(finish), sequence, steps, blocked)

    # --------------------------------------------------------------- request
    def request(self, process: str, req: list[int]) -> RequestResult:
        if process not in self.processes:
            return RequestResult(False, f"Unknown process {process}")
        i = self.processes.index(process)
        need = self.need[i]
        if any(r < 0 for r in req):
            return RequestResult(False, "Request entries must be non-negative")
        if any(r > n for r, n in zip(req, need)):
            return RequestResult(False, f"Error: {process} requested more than its declared maximum need {need}.")
        if any(r > a for r, a in zip(req, self.available)):
            return RequestResult(False, f"{process} must wait: not enough resources available {self.available}.")
        trial = self.copy()
        trial.available = [a - r for a, r in zip(trial.available, req)]
        trial.allocation[i] = [a + r for a, r in zip(trial.allocation[i], req)]
        res = trial.safety()
        if res.safe:
            return RequestResult(True, f"Granted. New state is safe: <{', '.join(res.sequence)}>.", res, trial)
        return RequestResult(
            False,
            f"Denied: granting would lead to an UNSAFE state (possible deadlock involving {', '.join(res.blocked)}). "
            f"{process} must wait.",
            res,
            trial,
        )

    def release(self, process: str, rel: list[int]) -> "BankersState":
        i = self.processes.index(process)
        if any(r > a for r, a in zip(rel, self.allocation[i])):
            raise ValueError("cannot release more than allocated")
        new = self.copy()
        new.allocation[i] = [a - r for a, r in zip(new.allocation[i], rel)]
        new.available = [a + r for a, r in zip(new.available, rel)]
        return new


def textbook_example() -> BankersState:
    """Silberschatz, Operating System Concepts - 5 processes, resources A(10) B(5) C(7)."""
    return BankersState(
        processes=["P0", "P1", "P2", "P3", "P4"],
        resources=["A", "B", "C"],
        available=[3, 3, 2],
        maximum=[[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]],
        allocation=[[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]],
    )


def unsafe_example() -> BankersState:
    return BankersState(
        processes=["P0", "P1", "P2"],
        resources=["R"],
        available=[2],
        maximum=[[10], [4], [9]],
        allocation=[[5], [2], [3]],
    )
