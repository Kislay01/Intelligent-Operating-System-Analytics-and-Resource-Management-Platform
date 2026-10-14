import random

import pytest

from kernelsense.paging import (
    ALL_POLICIES, BELADY_CLASSIC, PATTERNS, compare, detect_belady, fault_curve, generate, parse_refs, search_belady,
    working_set,
)

TEXTBOOK = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def test_textbook_fault_counts():
    faults = {r.algorithm: r.faults for r in compare(TEXTBOOK, 3)}
    assert faults["FIFO"] == 15
    assert faults["LRU"] == 12
    assert faults["Optimal"] == 9


def test_belady_classic():
    cases = detect_belady(BELADY_CLASSIC, max_frames=6)
    assert [(c.frames_low, c.faults_low, c.frames_high, c.faults_high) for c in cases] == [(3, 9, 4, 10)]
    assert detect_belady(BELADY_CLASSIC, policy="LRU") == []


def test_search_finds_anomaly():
    case = search_belady(seed=1)
    assert case is not None and case.faults_high > case.faults_low


@pytest.mark.parametrize("seed", range(20))
def test_optimal_is_lower_bound_and_stack_algorithms_are_monotone(seed):
    rng = random.Random(seed)
    refs = [rng.randrange(8) for _ in range(40)]
    for frames in range(1, 7):
        results = {r.algorithm: r.faults for r in compare(refs, frames)}
        assert results["Optimal"] == min(results.values())
    curve = fault_curve(refs, 7, ["LRU", "Optimal"])
    for algo in ("LRU", "Optimal"):
        f = curve[curve["algorithm"] == algo]["faults"].tolist()
        assert all(a >= b for a, b in zip(f, f[1:]))


def test_frames_never_exceed_capacity_and_hits_are_resident():
    refs = generate("locality", 60, 10)
    for name, cls in ALL_POLICIES.items():
        res = cls().simulate(refs, 4)
        prev = None
        for step in res.steps:
            assert len(step.frames) == 4
            resident = [f for f in step.frames if f is not None]
            assert len(resident) == len(set(resident))
            assert step.page in resident
            if step.hit:
                assert prev is not None and step.page in prev
            prev = resident


def test_compulsory_misses_only_when_frames_suffice():
    refs = generate("looping", 30, 10)  # loop over 5 pages
    assert ALL_POLICIES["LRU"]().simulate(refs, 5).faults == 5


def test_generators_and_parser():
    for p in PATTERNS:
        assert len(generate(p, 25, 10)) == (12 if p == "belady" else 25)
    assert parse_refs("1, 2 3") == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_refs("  ")


def test_working_set():
    assert working_set([1, 2, 1, 3, 3, 3], 3) == [1, 2, 2, 3, 2, 1]
