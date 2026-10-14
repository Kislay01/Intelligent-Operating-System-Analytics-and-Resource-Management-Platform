from datetime import datetime

from kernelsense import logs


def test_parse_syslog_iso_and_dmesg():
    r = logs.parse_line("Oct 14 11:52:01 host sshd[812]: Failed password for root from 1.2.3.4 port 22 ssh2", year=2026)
    assert r.timestamp == datetime(2026, 10, 14, 11, 52, 1)
    assert (r.host, r.source, r.pid, r.severity) == ("host", "sshd", 812, "err")

    r = logs.parse_line("2026-10-14T11:52:01.123456 host kernel: Out of memory: Killed process 42 (java)")
    assert r.source == "kernel" and r.severity == "crit" and r.timestamp.year == 2026

    boot = datetime(2026, 10, 14, 10, 0, 0)
    r = logs.parse_line("[   60.500000] usb 1-1: USB disconnect, device number 2", boot_time=boot)
    assert r.timestamp == datetime(2026, 10, 14, 10, 1, 0, 500000)
    assert r.severity == "notice"


def test_unknown_lines_are_kept():
    df = logs.parse_text("random text without format\n\n")
    assert len(df) == 1 and df.iloc[0]["source"] == "unknown"


def test_event_rules():
    cat, sev, groups = logs.classify_message("Out of memory: Killed process 4242 (python3) total-vm:1kB")
    assert cat == "OOM kill" and sev == "crit" and groups == {"pid": "4242", "proc": "python3"}
    assert logs.classify_message("worker[99]: segfault at 0 ip 0 sp 0 error 4")[0] == "Segmentation fault"
    assert logs.classify_message("INFO: task kworker:12 blocked for more than 120 seconds.")[0] == "Hung task"
    assert logs.classify_message("blk_update_request: I/O error, dev sda")[0] == "Disk I/O error"
    assert logs.classify_message("Started Session 3 of user kislay.")[0] is None


def test_sample_log_pipeline_detects_incidents():
    df = logs.parse_text(logs.sample_log(minutes=180, seed=42))
    assert len(df) > 500 and df["timestamp"].notna().all()
    events = logs.detect_events(df)
    assert {"OOM kill", "Authentication failure"} & set(events["category"])
    an = logs.detect_anomalies(df)
    assert an["anomaly"].sum() >= 1
    # the injected auth burst (25 lines in one minute) must be flagged
    burst = an.loc[an["lines"].idxmax()]
    assert bool(burst["anomaly"])


def test_load_sample_source():
    name, df = logs.load("sample")
    assert name == "sample" and not df.empty
