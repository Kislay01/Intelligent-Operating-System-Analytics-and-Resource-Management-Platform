"""Advanced page replacement: ARC, WSClock and Mattson's stack algorithm.

* ARC (Megiddo & Modha, FAST 2003) - Adaptive Replacement Cache. Keeps a
  recency list T1 and a frequency list T2 of resident pages plus "ghost" lists
  B1/B2 of recently evicted page numbers, and continuously adapts the target
  size p of T1: a hit in B1 means recency was undervalued, a hit in B2 means
  frequency was. It is scan-resistant, unlike LRU.
* WSClock (Carr & Hennessy) - Clock combined with the working-set model: a
  frame is only evicted when its reference bit is clear *and* it has not been
  used for more than tau references.
* Mattson et al. (1970) stack algorithm - for any stack algorithm (LRU), the
  page faults for *every* frame count can be computed in one pass from each
  reference's stack (reuse) distance: a reference with distance d hits iff the
  memory has >= d frames.
"""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from .algorithms import ALL_POLICIES, ReplacementPolicy, Step


class _Slots:
    """Maps resident pages to frame slots so traces can be shown as frame tables."""

    def __init__(self, n: int) -> None:
        self.slots: list[int | None] = [None] * n

    def place(self, page: int, victim: int | None) -> None:
        idx = self.slots.index(victim) if victim is not None else self.slots.index(None)
        self.slots[idx] = page

    def snapshot(self) -> list[int | None]:
        return list(self.slots)


class ARC(ReplacementPolicy):
    name = "ARC"

    def _run(self, refs, frames):
        c = frames
        t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        p = 0.0
        slots = _Slots(c)
        steps = []

        def replace(in_b2: bool) -> int:
            nonlocal p
            if t1 and (len(t1) > p or (in_b2 and len(t1) == int(p))):
                old, _ = t1.popitem(last=False)
                b1[old] = None
            else:
                old, _ = t2.popitem(last=False)
                b2[old] = None
            return old

        for i, x in enumerate(refs):
            if x in t1 or x in t2:  # Case I: hit
                (t1 if x in t1 else t2).pop(x)
                t2[x] = None
                steps.append(Step(i, x, True, slots.snapshot(), extra={"p": round(p, 2)}))
                continue
            victim = None
            if x in b1:  # Case II: ghost hit in B1 -> favour recency
                p = min(c, p + max(len(b2) / max(len(b1), 1), 1))
                victim = replace(False)
                b1.pop(x)
                t2[x] = None
            elif x in b2:  # Case III: ghost hit in B2 -> favour frequency
                p = max(0.0, p - max(len(b1) / max(len(b2), 1), 1))
                victim = replace(True)
                b2.pop(x)
                t2[x] = None
            else:  # Case IV: complete miss
                if len(t1) + len(b1) == c:
                    if len(t1) < c:
                        b1.popitem(last=False)
                        victim = replace(False)
                    else:
                        victim, _ = t1.popitem(last=False)
                elif len(t1) + len(b1) < c and len(t1) + len(t2) + len(b1) + len(b2) >= c:
                    if len(t1) + len(t2) + len(b1) + len(b2) == 2 * c:
                        b2.popitem(last=False)
                    if len(t1) + len(t2) >= c:
                        victim = replace(False)
                t1[x] = None
            slots.place(x, victim)
            steps.append(Step(i, x, False, slots.snapshot(), victim, extra={"p": round(p, 2)}))
        return steps


class WSClock(ReplacementPolicy):
    def __init__(self, tau: int = 8) -> None:
        self.tau = tau
        self.name = f"WSClock (tau={tau})"

    def _run(self, refs, frames):
        slots: list[int | None] = [None] * frames
        ref_bit = [0] * frames
        last_use = [0] * frames
        hand = 0
        steps = []
        for now, page in enumerate(refs):
            if page in slots:
                k = slots.index(page)
                ref_bit[k] = 1
                last_use[k] = now
                steps.append(Step(now, page, True, list(slots), extra={"hand": hand}))
                continue
            victim = None
            if None in slots:
                k = slots.index(None)
            else:
                k = None
                for _ in range(2 * frames):
                    if ref_bit[hand]:
                        ref_bit[hand] = 0
                        last_use[hand] = now
                    elif now - last_use[hand] > self.tau:
                        k = hand
                        break
                    hand = (hand + 1) % frames
                if k is None:  # every page is in the working set: evict the oldest
                    k = min(range(frames), key=lambda j: last_use[j])
                victim = slots[k]
                hand = (k + 1) % frames
            slots[k] = page
            ref_bit[k] = 1
            last_use[k] = now
            steps.append(Step(now, page, False, list(slots), victim, extra={"hand": hand}))
        return steps


def stack_distances(refs: list[int]) -> list[float]:
    """LRU stack distance of every reference (inf for first use)."""
    stack: list[int] = []  # most recent at the end
    out = []
    for x in refs:
        if x in stack:
            idx = stack.index(x)
            out.append(len(stack) - idx)
            stack.pop(idx)
        else:
            out.append(float("inf"))
        stack.append(x)
    return out


def mattson_lru_curve(refs: list[int], max_frames: int) -> pd.DataFrame:
    """LRU faults for every frame count 1..max_frames from a single pass."""
    dist = stack_distances(refs)
    n = len(refs)
    rows = []
    for c in range(1, max_frames + 1):
        faults = sum(1 for d in dist if d > c)
        rows.append({"frames": c, "faults": faults, "miss_ratio": round(faults / n, 4) if n else 0.0})
    return pd.DataFrame(rows)


def reuse_distance_histogram(refs: list[int]) -> pd.DataFrame:
    dist = stack_distances(refs)
    s = pd.Series(["cold" if d == float("inf") else int(d) for d in dist])
    return s.value_counts().rename_axis("distance").reset_index(name="count")


def register() -> None:
    """Add ARC and WSClock to the policy registry used by compare()/fault_curve()."""
    ALL_POLICIES.setdefault("ARC", ARC)
    ALL_POLICIES.setdefault("WSClock", WSClock)
