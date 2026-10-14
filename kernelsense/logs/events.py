"""Detection of OS-significant events in parsed logs.

Each rule maps a log signature to an event category and explains what it means
at the OS level, so the dashboard can say *why* a line matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EventRule:
    category: str
    severity: str
    pattern: re.Pattern
    meaning: str


RULES: list[EventRule] = [
    EventRule("OOM kill", "crit",
              re.compile(r"Out of memory: Kill(?:ed)? process (?P<pid>\d+) \((?P<proc>[^)]+)\)|oom-kill:|invoked oom-killer", re.I),
              "Kernel OOM killer terminated a process because RAM and swap were exhausted."),
    EventRule("Kernel panic / oops", "emerg",
              re.compile(r"Kernel panic|BUG: unable to handle|Oops:|general protection fault", re.I),
              "Kernel hit an unrecoverable or serious internal error."),
    EventRule("Soft/hard lockup", "crit",
              re.compile(r"soft lockup|hard LOCKUP|rcu_sched self-detected stall|rcu.*stall", re.I),
              "A CPU was stuck in kernel mode without scheduling for too long."),
    EventRule("Hung task", "warning",
              re.compile(r"task (?P<proc>\S+):(?P<pid>\d+) blocked for more than \d+ seconds|hung_task", re.I),
              "A task stayed in uninterruptible sleep (D state) too long - often stuck I/O or a kernel lock."),
    EventRule("Segmentation fault", "err",
              re.compile(r"(?P<proc>\S+)\[(?P<pid>\d+)\]: segfault at|segfault", re.I),
              "A process accessed memory it does not own (invalid page) and received SIGSEGV."),
    EventRule("Disk I/O error", "err",
              re.compile(r"I/O error|blk_update_request|Buffer I/O error|EXT4-fs error|read error on", re.I),
              "The block layer failed to read or write a device - possible failing disk."),
    EventRule("Thermal throttling", "warning",
              re.compile(r"temperature above threshold|cpu clock throttled|thermal.*throttl", re.I),
              "CPU frequency reduced to control temperature, lowering performance."),
    EventRule("Service failure", "err",
              re.compile(r"Failed to start|Main process exited, code=|entered failed state|Failed with result", re.I),
              "A systemd service failed or crashed."),
    EventRule("Authentication failure", "warning",
              re.compile(r"authentication failure|Failed password|Invalid user|FAILED SU", re.I),
              "A login or privilege escalation attempt failed."),
    EventRule("Device change", "notice",
              re.compile(r"USB disconnect|new (high|full|super)-speed USB device|Link is (Up|Down)|link becomes ready", re.I),
              "Hardware was attached, detached or a network link changed state."),
    EventRule("Memory pressure", "warning",
              re.compile(r"page allocation failure|low on memory|kswapd|swap.*full|Memory cgroup out of memory", re.I),
              "Kernel struggled to allocate memory or reclaim pages."),
]


def classify_message(message: str) -> tuple[str | None, str | None, dict]:
    for rule in RULES:
        m = rule.pattern.search(message)
        if m:
            return rule.category, rule.severity, {k: v for k, v in m.groupdict().items() if v}
    return None, None, {}


def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return only lines that match an OS event rule, with category and meaning."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "category", "severity", "source", "message", "meaning", "target"])
    meaning = {r.category: r.meaning for r in RULES}
    rows = []
    for rec in df.itertuples(index=False):
        cat, sev, groups = classify_message(rec.message)
        if cat:
            target = f"{groups.get('proc', '')}:{groups.get('pid', '')}".strip(":")
            rows.append({
                "timestamp": rec.timestamp, "category": cat, "severity": sev, "source": rec.source,
                "message": rec.message, "meaning": meaning[cat], "target": target,
            })
    return pd.DataFrame(rows)


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["category", "count", "first_seen", "last_seen"])
    g = events.groupby("category")
    return (
        pd.DataFrame({"count": g.size(), "first_seen": g["timestamp"].min(), "last_seen": g["timestamp"].max()})
        .reset_index()
        .sort_values("count", ascending=False)
    )
