"""Reference-string generators and paging analysis (Belady's anomaly, working set, thrashing)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .algorithms import FIFO, ALL_POLICIES

BELADY_CLASSIC = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]

PATTERNS = {
    "random": "Uniformly random pages (no locality)",
    "locality": "Hot set accessed most of the time (80/20 locality)",
    "looping": "A loop over a fixed set of pages (array traversal)",
    "sequential": "Sequential scan that never reuses pages (streaming / file read)",
    "phased": "Working set shifts between phases (function calls / program phases)",
    "belady": "Classic reference string exhibiting Belady's anomaly under FIFO",
}


def generate(pattern: str = "locality", length: int = 30, pages: int = 10, seed: int | None = 7) -> list[int]:
    rng = random.Random(seed)
    if pattern == "belady":
        return list(BELADY_CLASSIC)
    if pattern == "random":
        return [rng.randrange(pages) for _ in range(length)]
    if pattern == "looping":
        loop = max(2, pages // 2)
        return [i % loop for i in range(length)]
    if pattern == "sequential":
        return [i % pages for i in range(length)]
    if pattern == "phased":
        phases = 3
        ws_size = max(2, pages // phases)
        out = []
        for i in range(length):
            phase = min(i * phases // max(length, 1), phases - 1)
            base = (phase * ws_size) % pages
            out.append((base + rng.randrange(ws_size)) % pages)
        return out
    # locality: 80% of references go to 20% of pages
    hot = list(range(max(1, pages // 5)))
    cold = list(range(len(hot), pages)) or hot
    return [rng.choice(hot) if rng.random() < 0.8 else rng.choice(cold) for _ in range(length)]


def parse_refs(text: str) -> list[int]:
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("Reference string is empty")
    return [int(t) for t in tokens]


@dataclass
class BeladyCase:
    refs: list[int]
    frames_low: int
    faults_low: int
    frames_high: int
    faults_high: int


def detect_belady(refs: list[int], max_frames: int = 8, policy: str = "FIFO") -> list[BeladyCase]:
    """Find frame counts where adding a frame *increases* faults."""
    pol = ALL_POLICIES[policy]()
    faults = {f: pol.simulate(refs, f).faults for f in range(1, max_frames + 1)}
    return [
        BeladyCase(refs, f, faults[f], f + 1, faults[f + 1])
        for f in range(1, max_frames)
        if faults[f + 1] > faults[f]
    ]


def search_belady(length: int = 20, pages: int = 5, trials: int = 20000, seed: int = 1) -> BeladyCase | None:
    """Randomly search for a reference string that shows Belady's anomaly under FIFO."""
    rng = random.Random(seed)
    fifo = FIFO()
    for _ in range(trials):
        refs = [rng.randrange(1, pages + 1) for _ in range(length)]
        for f in range(1, pages):
            a, b = fifo.simulate(refs, f).faults, fifo.simulate(refs, f + 1).faults
            if b > a:
                return BeladyCase(refs, f, a, f + 1, b)
    return None


def working_set(refs: list[int], window: int) -> list[int]:
    """Working-set size W(t, delta) at each time t (Denning's model)."""
    return [len(set(refs[max(0, t - window + 1) : t + 1])) for t in range(len(refs))]


def thrashing_assessment(refs: list[int], frames: int, window: int = 10) -> dict:
    ws = working_set(refs, window)
    peak = max(ws) if ws else 0
    avg = sum(ws) / len(ws) if ws else 0
    return {
        "window": window,
        "avg_working_set": round(avg, 2),
        "peak_working_set": peak,
        "frames": frames,
        "thrashing_risk": "High" if avg > frames else ("Moderate" if peak > frames else "Low"),
        "explanation": (
            "When the working set is larger than the allocated frames, nearly every reference faults "
            "and the process spends more time paging than executing (thrashing)."
        ),
    }
