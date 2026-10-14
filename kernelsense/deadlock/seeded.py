"""Seeded lock-order bug benchmark for the lock-order validator.

Generates random multi-threaded programs over a set of locks. *Clean* programs
always acquire locks in a global order (deadlock-free by construction).
*Buggy* programs contain one path that acquires two locks against that order.

Each program is run twice:
* with threads executed one after another - a real deadlock cannot occur, so a
  timeout-based detector sees nothing; lockdep must predict the bug from order
  alone;
* concurrently with acquisition timeouts - to measure how often the bug
  actually manifests as a deadlock in a given run.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import pandas as pd

from .lockdep import LockOrderMonitor


@dataclass
class Program:
    buggy: bool
    paths: list[list[int]]  # each path = ordered list of lock indices one thread acquires (nested)


def generate(n_locks: int = 6, n_threads: int = 4, buggy: bool = False, rng: random.Random | None = None) -> Program:
    rng = rng or random.Random()
    paths = []
    for _ in range(n_threads):
        k = rng.randint(2, min(4, n_locks))
        paths.append(sorted(rng.sample(range(n_locks), k)))
    if buggy:
        # invert one pair on one path, using locks another path also takes in order
        victim = rng.randrange(n_threads)
        p = paths[victim]
        i, j = sorted(rng.sample(range(len(p)), 2))
        p[i], p[j] = p[j], p[i]
        # make sure some other path holds those two locks in the forward order
        other = (victim + 1) % n_threads
        lo, hi = min(p[i], p[j]), max(p[i], p[j])
        if not (lo in paths[other] and hi in paths[other]):
            paths[other] = sorted(set(paths[other]) | {lo, hi})
    return Program(buggy, paths)


def _run_path(locks, path, hold_s: float, timeout: float) -> bool:
    """Acquire locks along the path (nested). Returns False if an acquisition timed out."""
    taken = []
    try:
        for idx in path:
            if not locks[idx].acquire(timeout=timeout):
                return False
            taken.append(locks[idx])
            if hold_s:
                time.sleep(hold_s)
        return True
    finally:
        for lk in reversed(taken):
            lk.release()


def run_lockdep(program: Program) -> bool:
    mon = LockOrderMonitor()
    locks = [mon.lock(f"L{i}") for i in range(max(max(p) for p in program.paths) + 1)]
    for i, path in enumerate(program.paths):
        t = threading.Thread(target=_run_path, args=(locks, path, 0.0, 1.0), name=f"T{i}")
        t.start()
        t.join()  # sequential execution
    return bool(mon.violations)


def run_concurrent(program: Program, hold_s: float = 0.002, timeout: float = 0.2) -> bool:
    """True if a real deadlock (acquisition timeout) happened in this concurrent run."""
    n = max(max(p) for p in program.paths) + 1
    locks = [threading.Lock() for _ in range(n)]
    results = []

    def worker(path):
        results.append(_run_path(locks, path, hold_s, timeout))

    threads = [threading.Thread(target=worker, args=(p,)) for p in program.paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return not all(results)


def benchmark(n_programs: int = 200, seed: int = 0, concurrent_runs: int = 1, hold_s: float = 0.0) -> dict:
    """hold_s = time spent inside each critical section; 0 models short, realistic sections."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_programs):
        prog = generate(buggy=(i % 2 == 0), rng=rng)
        rows.append({
            "buggy": prog.buggy,
            "lockdep_flagged": run_lockdep(prog),
            "deadlock_observed": any(run_concurrent(prog, hold_s=hold_s) for _ in range(concurrent_runs)),
        })
    df = pd.DataFrame(rows)
    bug, clean = df[df.buggy], df[~df.buggy]
    return {
        "programs": n_programs,
        "hold_s": hold_s,
        "lockdep_detection_rate": float(bug.lockdep_flagged.mean()),
        "lockdep_false_positive_rate": float(clean.lockdep_flagged.mean()),
        "observed_deadlock_rate_buggy": float(bug.deadlock_observed.mean()),
        "observed_deadlock_rate_clean": float(clean.deadlock_observed.mean()),
        "table": df,
    }
