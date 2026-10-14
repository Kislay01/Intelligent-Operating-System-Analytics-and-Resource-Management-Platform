"""Property-based tests: invariants that must hold for *any* input."""

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from kernelsense.deadlock import BankersState  # noqa: E402
from kernelsense.paging import ALL_POLICIES, LRU, Optimal, mattson_lru_curve  # noqa: E402
from kernelsense.scheduling import CFS, EEVDF, FCFS, MLFQ, SJF, SRTF, Job, Lottery, RoundRobin, Stride  # noqa: E402

refs_strategy = st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=80)
jobs_strategy = st.lists(
    st.tuples(st.integers(0, 20), st.integers(1, 12), st.integers(0, 6)), min_size=1, max_size=8
).map(lambda xs: [Job(f"P{i}", a, b, p) for i, (a, b, p) in enumerate(xs)])


@settings(max_examples=60, deadline=None)
@given(refs_strategy, st.integers(1, 6))
def test_optimal_never_worse_than_any_policy(refs, frames):
    opt = Optimal().simulate(refs, frames).faults
    for cls in ALL_POLICIES.values():
        assert opt <= cls().simulate(refs, frames).faults


@settings(max_examples=60, deadline=None)
@given(refs_strategy)
def test_lru_is_a_stack_algorithm(refs):
    faults = [LRU().simulate(refs, f).faults for f in range(1, 8)]
    assert all(a >= b for a, b in zip(faults, faults[1:]))


@settings(max_examples=60, deadline=None)
@given(refs_strategy)
def test_mattson_equals_simulation(refs):
    assert mattson_lru_curve(refs, 6)["faults"].tolist() == [LRU().simulate(refs, f).faults for f in range(1, 7)]


@settings(max_examples=40, deadline=None)
@given(jobs_strategy)
def test_every_scheduler_conserves_work_and_respects_arrivals(jobs):
    for sched in (FCFS(), SJF(), SRTF(), RoundRobin(3), MLFQ(), CFS(), EEVDF(), Stride(), Lottery()):
        res = sched.run(jobs)
        ran = {}
        for seg in res.gantt:
            assert seg.end > seg.start
            if seg.pid != "IDLE":
                ran[seg.pid] = ran.get(seg.pid, 0) + seg.length
        assert ran == {j.pid: j.burst for j in jobs}, sched.name
        by = {j.pid: j for j in jobs}
        for seg in res.gantt:
            if seg.pid != "IDLE":
                assert seg.start >= by[seg.pid].arrival
        # no two segments overlap on the single CPU
        segs = sorted(res.gantt, key=lambda s: s.start)
        assert all(a.end <= b.start for a, b in zip(segs, segs[1:]))


@settings(max_examples=40, deadline=None)
@given(jobs_strategy)
def test_srtf_minimises_average_waiting_among_policies(jobs):
    best = SRTF().run(jobs).summary()["avg_waiting"]
    for sched in (FCFS(), RoundRobin(2), CFS(), Stride()):
        assert best <= sched.run(jobs).summary()["avg_waiting"] + 1e-9


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 4), st.integers(1, 3), st.data())
def test_bankers_granted_requests_keep_state_safe(n, m, data):
    maximum = [[data.draw(st.integers(0, 5)) for _ in range(m)] for _ in range(n)]
    allocation = [[data.draw(st.integers(0, maximum[i][j])) for j in range(m)] for i in range(n)]
    available = [data.draw(st.integers(0, 4)) for _ in range(m)]
    state = BankersState([f"P{i}" for i in range(n)], [chr(65 + j) for j in range(m)], available, maximum, allocation)
    i = data.draw(st.integers(0, n - 1))
    req = [data.draw(st.integers(0, max(0, state.need[i][j]))) for j in range(m)]
    r = state.request(f"P{i}", req)
    if r.granted:
        assert r.new_state.safety().safe
        assert all(a >= 0 for a in r.new_state.available)
