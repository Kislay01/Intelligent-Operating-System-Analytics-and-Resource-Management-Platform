"""Page replacement algorithms.

Each policy implements ``ReplacementPolicy.simulate(refs, frames)`` and returns a
step-by-step trace so the UI can show the frame table, hits, misses and the
victim page at every reference.

Implemented: FIFO, LRU, Optimal (Belady's MIN), LFU, Clock (Second Chance).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Step:
    index: int
    page: int
    hit: bool
    frames: list[int | None]
    evicted: int | None = None
    extra: dict = field(default_factory=dict)  # e.g. reference bits for Clock


@dataclass
class PagingResult:
    algorithm: str
    frames: int
    steps: list[Step]

    @property
    def faults(self) -> int:
        return sum(not s.hit for s in self.steps)

    @property
    def hits(self) -> int:
        return sum(s.hit for s in self.steps)

    @property
    def fault_rate(self) -> float:
        return self.faults / len(self.steps) if self.steps else 0.0

    def summary(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "frames": self.frames,
            "references": len(self.steps),
            "faults": self.faults,
            "hits": self.hits,
            "fault_rate": round(self.fault_rate, 4),
            "hit_ratio": round(1 - self.fault_rate, 4),
        }

    def table(self) -> pd.DataFrame:
        """Frame table: one column per reference, one row per frame (textbook layout)."""
        data = {}
        for s in self.steps:
            col = f"{s.index + 1}:{s.page}"
            data[col] = ["" if f is None else f for f in s.frames] + ["H" if s.hit else "F"]
        idx = [f"Frame {i}" for i in range(self.frames)] + ["Result"]
        return pd.DataFrame(data, index=idx)


class ReplacementPolicy(ABC):
    name = "Policy"

    def simulate(self, refs: list[int], frames: int) -> PagingResult:
        if frames <= 0:
            raise ValueError("frames must be positive")
        return PagingResult(self.name, frames, self._run(list(refs), frames))

    @abstractmethod
    def _run(self, refs: list[int], frames: int) -> list[Step]: ...


def _snapshot(slots: list[int | None]) -> list[int | None]:
    return list(slots)


class FIFO(ReplacementPolicy):
    name = "FIFO"

    def _run(self, refs, frames):
        slots: list[int | None] = [None] * frames
        order: deque[int] = deque()  # slot indices in load order
        steps = []
        for i, page in enumerate(refs):
            if page in slots:
                steps.append(Step(i, page, True, _snapshot(slots)))
                continue
            evicted = None
            if None in slots:
                idx = slots.index(None)
            else:
                idx = order.popleft()
                evicted = slots[idx]
            slots[idx] = page
            order.append(idx)
            steps.append(Step(i, page, False, _snapshot(slots), evicted))
        return steps


class LRU(ReplacementPolicy):
    name = "LRU"

    def _run(self, refs, frames):
        slots: list[int | None] = [None] * frames
        recency: OrderedDict[int, None] = OrderedDict()  # least recent first
        steps = []
        for i, page in enumerate(refs):
            if page in recency:
                recency.move_to_end(page)
                steps.append(Step(i, page, True, _snapshot(slots)))
                continue
            evicted = None
            if None in slots:
                idx = slots.index(None)
            else:
                evicted, _ = recency.popitem(last=False)
                idx = slots.index(evicted)
            slots[idx] = page
            recency[page] = None
            steps.append(Step(i, page, False, _snapshot(slots), evicted))
        return steps


class Optimal(ReplacementPolicy):
    """Belady's MIN: evict the page whose next use is farthest in the future.

    Not implementable in a real OS (needs future knowledge); used as the lower
    bound every other policy is compared against.
    """

    name = "Optimal"

    def _run(self, refs, frames):
        # Precompute next-use positions in O(n).
        n = len(refs)
        next_use = [n] * n
        last_seen: dict[int, int] = {}
        for i in range(n - 1, -1, -1):
            next_use[i] = last_seen.get(refs[i], float("inf"))
            last_seen[refs[i]] = i
        slots: list[int | None] = [None] * frames
        upcoming: dict[int, float] = {}
        steps = []
        for i, page in enumerate(refs):
            if page in slots:
                upcoming[page] = next_use[i]
                steps.append(Step(i, page, True, _snapshot(slots)))
                continue
            evicted = None
            if None in slots:
                idx = slots.index(None)
            else:
                evicted = max((p for p in slots if p is not None), key=lambda p: (upcoming[p], -slots.index(p)))
                idx = slots.index(evicted)
                upcoming.pop(evicted, None)
            slots[idx] = page
            upcoming[page] = next_use[i]
            steps.append(Step(i, page, False, _snapshot(slots), evicted))
        return steps


class LFU(ReplacementPolicy):
    """Least Frequently Used; ties broken by least recently used."""

    name = "LFU"

    def _run(self, refs, frames):
        slots: list[int | None] = [None] * frames
        freq: dict[int, int] = {}
        last: dict[int, int] = {}
        steps = []
        for i, page in enumerate(refs):
            if page in slots:
                freq[page] += 1
                last[page] = i
                steps.append(Step(i, page, True, _snapshot(slots), extra={"freq": dict(freq)}))
                continue
            evicted = None
            if None in slots:
                idx = slots.index(None)
            else:
                evicted = min((p for p in slots if p is not None), key=lambda p: (freq[p], last[p]))
                idx = slots.index(evicted)
                freq.pop(evicted)
                last.pop(evicted)
            slots[idx] = page
            freq[page] = 1
            last[page] = i
            steps.append(Step(i, page, False, _snapshot(slots), evicted, extra={"freq": dict(freq)}))
        return steps


class Clock(ReplacementPolicy):
    """Second-chance (Clock): a circular buffer with one reference bit per frame.

    On a fault the hand sweeps; frames with bit=1 get a second chance (bit
    cleared), the first frame with bit=0 is the victim. Approximates LRU with
    the single hardware-maintained accessed bit real MMUs provide.
    """

    name = "Clock"

    def _run(self, refs, frames):
        slots: list[int | None] = [None] * frames
        ref_bit = [0] * frames
        hand = 0
        steps = []
        for i, page in enumerate(refs):
            if page in slots:
                ref_bit[slots.index(page)] = 1
                steps.append(Step(i, page, True, _snapshot(slots), extra={"bits": list(ref_bit), "hand": hand}))
                continue
            evicted = None
            if None in slots:
                idx = slots.index(None)
            else:
                while ref_bit[hand] == 1:
                    ref_bit[hand] = 0
                    hand = (hand + 1) % frames
                idx = hand
                evicted = slots[idx]
            slots[idx] = page
            ref_bit[idx] = 1
            hand = (idx + 1) % frames
            steps.append(Step(i, page, False, _snapshot(slots), evicted, extra={"bits": list(ref_bit), "hand": hand}))
        return steps


ALL_POLICIES: dict[str, type[ReplacementPolicy]] = {
    "FIFO": FIFO,
    "LRU": LRU,
    "Optimal": Optimal,
    "LFU": LFU,
    "Clock": Clock,
}


def compare(refs: list[int], frames: int, policies: list[str] | None = None) -> list[PagingResult]:
    names = policies or list(ALL_POLICIES)
    return [ALL_POLICIES[n]().simulate(refs, frames) for n in names]


def fault_curve(refs: list[int], max_frames: int, policies: list[str] | None = None) -> pd.DataFrame:
    """Page faults for every frame count 1..max_frames, for each policy."""
    names = policies or list(ALL_POLICIES)
    rows = []
    for f in range(1, max_frames + 1):
        for n in names:
            r = ALL_POLICIES[n]().simulate(refs, f)
            rows.append({"frames": f, "algorithm": n, "faults": r.faults, "fault_rate": round(r.fault_rate, 4)})
    return pd.DataFrame(rows)
