"""Log ingestion, parsing, OS event detection and anomaly detection."""

from .anomaly import detect_anomalies, top_sources, window_features
from .events import RULES, classify_message, detect_events, event_summary
from .parser import SEVERITY_ORDER, LogRecord, infer_severity, parse_line, parse_text
from .sources import available_sources, boot_time, read_dmesg, read_journal, read_syslog_file, sample_log


def load(source: str = "auto", lines: int = 5000, path: str | None = None, min_lines: int = 20):
    """Load and parse logs from a named source.

    'auto' tries journald, syslog, dmesg and falls back to the generated sample
    log when no real source has at least ``min_lines`` lines.
    """
    readers = {
        "journald": lambda: (read_journal(lines), None),
        "syslog": lambda: (read_syslog_file(path, lines), None),
        "dmesg": lambda: (read_dmesg(lines), boot_time()),
        "sample": lambda: (sample_log(), None),
    }
    order = ["journald", "syslog", "dmesg", "sample"] if source == "auto" else [source]
    for name in order:
        text, bt = readers[name]()
        enough = len(text.splitlines()) >= (min_lines if source == "auto" and name != "sample" else 1)
        if text.strip() and enough:
            return name, parse_text(text, boot_time=bt)
    return "none", parse_text("")


__all__ = [
    "LogRecord", "RULES", "SEVERITY_ORDER", "available_sources", "boot_time", "classify_message",
    "detect_anomalies", "detect_events", "event_summary", "infer_severity", "load", "parse_line", "parse_text",
    "read_dmesg", "read_journal", "read_syslog_file", "sample_log", "top_sources", "window_features",
]
