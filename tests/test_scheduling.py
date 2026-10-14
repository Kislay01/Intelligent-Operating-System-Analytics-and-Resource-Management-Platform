import pytest

from kernelsense.scheduling import (
    FCFS, MLFQ, SJF, SRTF, Job, PriorityScheduler, RoundRobin, best_alpha, exponential_average, recommend,
)
from kernelsense.scheduling.workload import PROFILES, from_csv, synthetic

CONVOY = [Job("P1", 0, 24), Job("P2", 0, 3), Job("P3", 0, 3)]
SRTF_CASE = [Job("P1", 0, 8), Job("P2", 1, 4), Job("P3", 2, 9), Job("P4", 3, 5)]
PRIORITY_CASE = [Job("P1", 0, 10, 3), Job("P2", 0, 1, 1), Job("P3", 0, 2, 4), Job("P4", 0, 1, 5), Job("P5", 0, 5, 2)]


# Expected values from Silberschatz, Operating System Concepts (ch. 5).
@pytest.mark.parametrize("sched,expected", [(FCFS(), 17.0), (SJF(), 3.0), (RoundRobin(4), 5.67)])
def test_textbook_average_waiting(sched, expected):
    assert sched.run(CONVOY).summary()["avg_waiting"] == expected


def test_srtf_textbook():
    assert SRTF().run(SRTF_CASE).summary()["avg_waiting"] == 6.5


def test_priority_non_preemptive_textbook():
    assert PriorityScheduler(preemptive=False).run(PRIORITY_CASE).summary()["avg_waiting"] == 8.2


@pytest.mark.parametrize("sched", [FCFS(), SJF(), SRTF(), RoundRobin(3), PriorityScheduler(True, 5), MLFQ()])
def test_every_job_runs_exactly_its_burst(sched):
    jobs = synthetic("balanced", 10, seed=3)
    res = sched.run(jobs)
    ran = {}
    for seg in res.gantt:
        if seg.pid != "IDLE":
            ran[seg.pid] = ran.get(seg.pid, 0) + seg.length
    assert ran == {j.pid: j.burst for j in jobs}
    for s in res.stats:
        assert s.waiting >= 0 and s.response >= 0 and s.completion >= s.arrival + s.burst


def test_idle_gap_is_recorded():
    res = FCFS().run([Job("A", 0, 2), Job("B", 5, 1)])
    assert [(g.pid, g.start, g.end) for g in res.gantt] == [("A", 0, 2), ("IDLE", 2, 5), ("B", 5, 6)]


def test_sjf_is_optimal_for_waiting_when_all_arrive_together():
    jobs = [Job(f"P{i}", 0, b) for i, b in enumerate([6, 8, 7, 3])]
    sjf = SJF().run(jobs).summary()["avg_waiting"]
    for s in (FCFS(), RoundRobin(2), RoundRobin(5)):
        assert sjf <= s.run(jobs).summary()["avg_waiting"]


def test_mlfq_demotes_long_jobs():
    res = MLFQ((2, 4), boost_period=None).run([Job("long", 0, 12), Job("short", 0, 1)])
    levels = {g.queue for g in res.gantt if g.pid == "long"}
    assert levels == {0, 1, 2}


def test_aging_prevents_starvation():
    jobs = [Job("low", 0, 3, priority=10)] + [Job(f"h{i}", i * 2, 2, priority=0) for i in range(15)]
    no_aging = PriorityScheduler(True).run(jobs)
    aging = PriorityScheduler(True, aging_interval=2).run(jobs)
    low = lambda r: next(s for s in r.stats if s.pid == "low").completion
    assert low(aging) < low(no_aging)


def test_context_switch_count():
    assert FCFS().run(CONVOY).context_switches == 2
    assert RoundRobin(4).run(CONVOY).context_switches == 2 + 1  # P1 P2 P3 P1 then P1 runs alone


def test_invalid_jobs():
    with pytest.raises(ValueError):
        Job("x", 0, 0)
    with pytest.raises(ValueError):
        FCFS().run([Job("a", 0, 1), Job("a", 1, 1)])


def test_exponential_average_textbook():
    trace = exponential_average([6, 4, 6, 4, 13, 13, 13], alpha=0.5, tau0=10)
    assert trace.predicted == [10, 8, 6, 6, 5, 9, 11]
    assert trace.next_prediction == 12
    alpha, mae = best_alpha([5, 5, 5, 5])
    assert mae == 0


@pytest.mark.parametrize("profile", list(PROFILES))
def test_recommender_runs_for_every_profile_and_goal(profile):
    jobs = synthetic(profile, 8)
    for goal in ("balanced", "interactive", "throughput", "fairness"):
        r = recommend(jobs, goal)
        assert r.best == r.ranking.iloc[0]["algorithm"]
        assert r.reasons
    practical = recommend(jobs, "interactive", practical_only=True)
    assert not practical.ranking["algorithm"].isin(["SJF", "SRTF"]).any()


def test_recommender_flags_convoy():
    r = recommend(synthetic("convoy", 8), "interactive")
    assert r.profile.convoy_risk
    assert r.best != "FCFS"


def test_csv_loader():
    jobs = from_csv("pid,arrival,burst,priority\nA,0,5,1\nB,2,3,0\n")
    assert [(j.pid, j.burst, j.priority) for j in jobs] == [("A", 5, 1), ("B", 3, 0)]
