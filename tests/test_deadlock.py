import pytest

from kernelsense.deadlock import (
    BankersState, ResourceAllocationGraph, bankers_risk, dining_philosophers, graph_risk, multi_instance_no_deadlock,
    parse_proc_locks, simple_deadlock, textbook_example, unsafe_example,
)
from kernelsense.deadlock.live import file_lock_wait_for


def test_textbook_safe_sequence():
    res = textbook_example().safety()
    assert res.safe
    assert res.sequence == ["P1", "P3", "P4", "P0", "P2"]


def test_textbook_requests():
    s = textbook_example()
    r1 = s.request("P1", [1, 0, 2])
    assert r1.granted
    s2 = r1.new_state
    assert s2.available == [2, 3, 0]
    assert not s2.request("P4", [3, 3, 0]).granted  # not enough available
    r3 = s2.request("P0", [0, 2, 0])
    assert not r3.granted and "UNSAFE" in r3.reason
    assert not s.request("P1", [5, 5, 5]).granted  # exceeds declared need


def test_unsafe_example_and_risk_ordering():
    assert not unsafe_example().safety().safe
    assert bankers_risk(unsafe_example()).score > bankers_risk(textbook_example()).score
    assert bankers_risk(unsafe_example()).level in ("High", "Critical")


def test_release_restores_resources():
    s = textbook_example()
    s2 = s.release("P2", [3, 0, 2])
    assert s2.available == [6, 3, 4]
    with pytest.raises(ValueError):
        s.release("P0", [9, 9, 9])


def test_invalid_state_rejected():
    with pytest.raises(ValueError):
        BankersState(["P0"], ["A"], [1], [[1]], [[2]])  # allocation > max


def test_single_instance_cycle_is_deadlock():
    d = simple_deadlock().detect()
    assert d.deadlocked and set(d.deadlocked_processes) == {"P1", "P2"}
    d = dining_philosophers(5).detect()
    assert d.deadlocked and len(d.deadlocked_processes) == 5
    assert graph_risk(simple_deadlock()).score == 100


def test_cycle_without_deadlock_in_multi_instance_graph():
    g = multi_instance_no_deadlock()
    d = g.detect()
    assert not d.deadlocked and d.cycles
    assert set(d.finish_order) == {"P1", "P2", "P3", "P4"}


def test_no_cycle_no_deadlock():
    g = ResourceAllocationGraph()
    g.add_resource("R1").add_resource("R2")
    g.assign("R1", "P1").request("P2", "R1")
    d = g.detect()
    assert not d.deadlocked
    with pytest.raises(ValueError):
        g.assign("R1", "P3")  # only one instance


def test_proc_locks_parser_and_cycle():
    text = (
        "1: POSIX  ADVISORY  WRITE 100 08:01:1 0 EOF\n"
        "1: -> POSIX  ADVISORY  WRITE 200 08:01:1 0 EOF\n"
        "2: FLOCK  ADVISORY  WRITE 200 08:01:2 0 EOF\n"
        "2: -> FLOCK  ADVISORY  WRITE 100 08:01:2 0 EOF\n"
    )
    locks = parse_proc_locks(text)
    assert len(locks) == 4 and sum(l.blocked for l in locks) == 2
    g = file_lock_wait_for(locks)
    assert set(g.edges()) == {(200, 100), (100, 200)}
