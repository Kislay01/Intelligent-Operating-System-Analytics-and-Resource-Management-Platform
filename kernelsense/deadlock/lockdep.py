"""Lock-order validator inspired by the Linux kernel's lockdep.

Detecting a deadlock after it happens is easy: the program hangs. lockdep
predicts deadlocks that *have not happened yet*. Every time a thread acquires
lock B while holding lock A, the edge A -> B is added to a global lock-order
graph. If an acquisition would add an edge that closes a cycle (somewhere,
possibly in another thread at another time, B was taken before A), there is an
interleaving in which the threads wait for each other forever, so a potential
deadlock is reported immediately - even though this particular run is fine.

As in the kernel, locks are grouped into *classes*: by default a lock's class
is the source line that created it, so every instance created by the same line
shares ordering rules (``patch_threading`` does this for unmodified code).
"""

from __future__ import annotations

import contextlib
import threading
import traceback
from dataclasses import dataclass, field

import networkx as nx

_REAL_LOCK = threading.Lock
_REAL_RLOCK = threading.RLock


@dataclass
class LockOrderViolation:
    acquiring: str
    held: str
    thread: str
    cycle: list[str]
    site: str
    previous_sites: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (f"Potential deadlock: thread '{self.thread}' takes {self.acquiring} while holding {self.held}, "
                f"but the opposite order was seen before (cycle {' -> '.join(self.cycle + [self.cycle[0]])}).")


def _caller_site(skip: int = 3) -> str:
    frames = traceback.extract_stack(limit=skip + 1)
    f = frames[0]
    return f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}"


class LockOrderMonitor:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.violations: list[LockOrderViolation] = []
        self._held = threading.local()
        self._meta = _REAL_LOCK()
        self.acquisitions = 0
        self._seen_pairs: set[tuple[str, str]] = set()

    # --------------------------------------------------------------- factory
    def lock(self, name: str | None = None, reentrant: bool = False) -> "TrackedLock":
        return TrackedLock(self, name or f"lock@{_caller_site(2)}", reentrant)

    def _stack(self) -> list[str]:
        if not hasattr(self._held, "stack"):
            self._held.stack = []
        return self._held.stack

    # ------------------------------------------------------------ bookkeeping
    def before_acquire(self, name: str) -> None:
        stack = self._stack()
        thread = threading.current_thread().name
        site = _caller_site(4)
        with self._meta:
            self.acquisitions += 1
            self.graph.add_node(name)
            for held in dict.fromkeys(stack):  # unique, in order
                if held == name:
                    continue
                if (held, name) not in self._seen_pairs and self.graph.has_node(name) and nx.has_path(self.graph, name, held):
                    cycle = nx.shortest_path(self.graph, name, held)
                    prev = [self.graph.edges[u, v]["site"] for u, v in zip(cycle, cycle[1:])]
                    self.violations.append(LockOrderViolation(name, held, thread, cycle, site, prev))
                if not self.graph.has_edge(held, name):
                    self.graph.add_edge(held, name, site=site, thread=thread)
                self._seen_pairs.add((held, name))

    def after_acquire(self, name: str) -> None:
        self._stack().append(name)

    def on_release(self, name: str) -> None:
        stack = self._stack()
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] == name:
                del stack[i]
                break

    # ---------------------------------------------------------------- report
    def report(self) -> dict:
        return {
            "locks": self.graph.number_of_nodes(),
            "order_edges": self.graph.number_of_edges(),
            "acquisitions": self.acquisitions,
            "violations": len(self.violations),
            "details": [v.describe() for v in self.violations],
        }

    def reset(self) -> None:
        with self._meta:
            self.graph.clear()
            self.violations.clear()
            self._seen_pairs.clear()
            self.acquisitions = 0


class TrackedLock:
    """Drop-in replacement for threading.Lock / RLock that reports lock ordering."""

    def __init__(self, monitor: LockOrderMonitor, name: str, reentrant: bool = False) -> None:
        self.monitor = monitor
        self.name = name
        self._lock = _REAL_RLOCK() if reentrant else _REAL_LOCK()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        self.monitor.before_acquire(self.name)
        ok = self._lock.acquire(blocking, timeout)
        if ok:
            self.monitor.after_acquire(self.name)
        return ok

    def release(self) -> None:
        self.monitor.on_release(self.name)
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked() if hasattr(self._lock, "locked") else False

    __enter__ = acquire

    def __exit__(self, *exc) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"<TrackedLock {self.name}>"


@contextlib.contextmanager
def patch_threading(monitor: LockOrderMonitor):
    """Temporarily make threading.Lock/RLock return tracked locks (class = creation site).

    Lets unmodified code be validated:
        mon = LockOrderMonitor()
        with patch_threading(mon):
            run_my_program()
        print(mon.report())
    """
    def make_lock():
        return TrackedLock(monitor, f"Lock@{_caller_site(2)}")

    def make_rlock():
        return TrackedLock(monitor, f"RLock@{_caller_site(2)}", reentrant=True)

    threading.Lock, threading.RLock = make_lock, make_rlock
    try:
        yield monitor
    finally:
        threading.Lock, threading.RLock = _REAL_LOCK, _REAL_RLOCK


# --------------------------------------------------------------------------- #
# Demonstrations
# --------------------------------------------------------------------------- #
def demo_inversion() -> LockOrderMonitor:
    """Two threads take the same two locks in opposite order - but at different
    times, so the program never actually deadlocks. lockdep still reports it."""
    mon = LockOrderMonitor()
    a, b = mon.lock("accounts_lock"), mon.lock("audit_lock")

    def transfer():
        with a:
            with b:
                pass

    def audit():
        with b:
            with a:
                pass

    t1 = threading.Thread(target=transfer, name="transfer")
    t1.start()
    t1.join()  # sequential: no real deadlock can happen in this run
    t2 = threading.Thread(target=audit, name="audit")
    t2.start()
    t2.join()
    return mon


def demo_dining_philosophers(n: int = 5, ordered: bool = False) -> LockOrderMonitor:
    """Philosophers pick up left then right fork. Without a global order the
    fork graph forms a cycle; with resource ordering (lower fork first) it does not."""
    mon = LockOrderMonitor()
    forks = [mon.lock(f"fork{i}") for i in range(n)]

    def dine(i):
        left, right = forks[i], forks[(i + 1) % n]
        first, second = (left, right) if not ordered or i < (i + 1) % n else (right, left)
        with first:
            with second:
                pass

    for i in range(n):
        t = threading.Thread(target=dine, args=(i,), name=f"philosopher{i}")
        t.start()
        t.join()
    return mon
