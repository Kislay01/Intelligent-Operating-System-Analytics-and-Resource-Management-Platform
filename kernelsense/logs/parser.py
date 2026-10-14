"""Parsers for Linux log formats.

Supported line formats:
    syslog (RFC 3164):   Oct 14 11:52:01 host sshd[812]: message
    ISO / journalctl:    2026-10-14T11:52:01.123456+05:30 host kernel: message
    dmesg:               [  123.456789] message        (seconds since boot)
    dmesg --time-format iso / -T variants are handled by the ISO pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

SYSLOG_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<source>[^\s:\[]+)(?:\[(?P<pid>\d+)\])?:\s?(?P<msg>.*)$"
)
ISO_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?P<host>\S+)\s+(?P<source>[^\s:\[]+)(?:\[(?P<pid>\d+)\])?:\s?(?P<msg>.*)$"
)
DMESG_RE = re.compile(r"^\[\s*(?P<uptime>\d+\.\d+)\]\s(?P<msg>.*)$")

MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

SEVERITY_ORDER = ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"]
_SEVERITY_PATTERNS = [
    ("emerg", re.compile(r"kernel panic|panic -|emergency", re.I)),
    ("crit", re.compile(r"\bcritical\b|\bcrit\b|oops|bug:|call trace|soft lockup|hard lockup|out of memory|oom-kill", re.I)),
    ("err", re.compile(r"\berror\b|\bfailed\b|failure|segfault|i/o error|cannot|unable to|denied|fatal", re.I)),
    ("warning", re.compile(r"\bwarn(ing)?\b|throttl|deprecated|timeout|timed out|retry|blocked for more than", re.I)),
    ("notice", re.compile(r"\bnotice\b|disconnect|reconnect|link (is )?(up|down)", re.I)),
    ("debug", re.compile(r"\bdebug\b", re.I)),
]


@dataclass
class LogRecord:
    timestamp: datetime | None
    host: str
    source: str
    pid: int | None
    message: str
    severity: str
    raw: str


def infer_severity(source: str, message: str) -> str:
    for level, pattern in _SEVERITY_PATTERNS:
        if pattern.search(message):
            return level
    return "info"


def parse_line(line: str, year: int | None = None, boot_time: datetime | None = None) -> LogRecord | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None
    m = ISO_RE.match(line)
    if m:
        try:
            ts = pd.to_datetime(m["ts"].replace(",", "."), utc=False).to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.astimezone().replace(tzinfo=None)
        except (ValueError, TypeError):
            ts = None
        src = m["source"]
        return LogRecord(ts, m["host"], src, int(m["pid"]) if m["pid"] else None, m["msg"],
                         infer_severity(src, m["msg"]), line)
    m = SYSLOG_RE.match(line)
    if m:
        y = year or datetime.now().year
        try:
            ts = datetime.strptime(f"{y}-{MONTHS[m['mon']]:02d}-{int(m['day']):02d} {m['time']}", "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            ts = None
        src = m["source"]
        return LogRecord(ts, m["host"], src, int(m["pid"]) if m["pid"] else None, m["msg"],
                         infer_severity(src, m["msg"]), line)
    m = DMESG_RE.match(line)
    if m:
        ts = boot_time + timedelta(seconds=float(m["uptime"])) if boot_time else None
        return LogRecord(ts, "localhost", "kernel", None, m["msg"], infer_severity("kernel", m["msg"]), line)
    # Unknown format: keep it, so nothing is silently dropped.
    return LogRecord(None, "", "unknown", None, line, infer_severity("", line), line)


def parse_text(text: str, year: int | None = None, boot_time: datetime | None = None) -> pd.DataFrame:
    records = [r for r in (parse_line(l, year, boot_time) for l in text.splitlines()) if r]
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "host", "source", "pid", "message", "severity", "raw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["severity"] = pd.Categorical(df["severity"], categories=SEVERITY_ORDER, ordered=True)
    return df
