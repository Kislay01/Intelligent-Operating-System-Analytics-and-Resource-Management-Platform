"""Log sources: journald, syslog files, the kernel ring buffer, or a sample log.

Every reader returns raw text so the same parser handles all of them. Readers
fail softly (empty string) when a source is missing or not permitted, which is
common on WSL without systemd or when running without root.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import psutil

SYSLOG_FILES = ["/var/log/syslog", "/var/log/messages", "/var/log/kern.log"]


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if out.returncode != 0:
            return ""
        # journalctl prints markers such as "-- No entries --" / "-- Boot ... --"
        return "\n".join(l for l in out.stdout.splitlines() if not l.startswith("-- "))
    except (OSError, subprocess.SubprocessError):
        return ""


def read_journal(lines: int = 5000, since: str | None = None, kernel_only: bool = False) -> str:
    cmd = ["journalctl", "--no-pager", "-o", "short-iso", "-n", str(lines)]
    if since:
        cmd += ["--since", since]
    if kernel_only:
        cmd.append("-k")
    return _run(cmd)


def read_dmesg(lines: int = 5000) -> str:
    # Raw format "[seconds-since-boot] message"; converted to wall time with the boot time.
    text = _run(["dmesg"])
    return "\n".join(text.splitlines()[-lines:])


def read_syslog_file(path: str | None = None, lines: int = 5000) -> str:
    candidates = [path] if path else SYSLOG_FILES
    for p in candidates:
        try:
            with open(p, "r", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError:
            continue
    return ""


def boot_time() -> datetime:
    return datetime.fromtimestamp(psutil.boot_time())


def available_sources() -> dict[str, bool]:
    return {
        "journald": bool(shutil.which("journalctl")) and bool(read_journal(lines=1)),
        "dmesg": bool(read_dmesg(lines=1)),
        "syslog": any(Path(p).exists() for p in SYSLOG_FILES),
        "sample": True,
    }


# --------------------------------------------------------------------------- #
# Sample log generator - realistic syslog with injected incidents, used for
# demos and tests on machines where real logs are unavailable.
# --------------------------------------------------------------------------- #
_NORMAL = [
    ("systemd", "Started Session {n} of user kislay."),
    ("systemd", "Starting Daily apt download activities..."),
    ("CRON", "(root) CMD (command -v debian-sa1 > /dev/null && debian-sa1 1 1)"),
    ("sshd", "Accepted publickey for kislay from 192.168.1.{n} port {p} ssh2"),
    ("kernel", "e1000: eth0: Rx/Tx flow control: None"),
    ("NetworkManager", "<info>  [{n}.{p}] device (eth0): state change: activated"),
    ("dbus-daemon", "[system] Successfully activated service 'org.freedesktop.hostname1'"),
    ("systemd-resolved", "Using degraded feature set UDP instead of TCP for DNS server 8.8.8.8."),
    ("rsyslogd", "[origin software=\"rsyslogd\"] rsyslogd was HUPed"),
    ("python3", "worker {n} finished batch in {p} ms"),
]
_INCIDENTS = {
    "oom": [
        ("kernel", "python3 invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0"),
        ("kernel", "Out of memory: Killed process {pid} (python3) total-vm:8123456kB, anon-rss:6234567kB"),
        ("kernel", "oom_reaper: reaped process {pid} (python3), now anon-rss:0kB"),
    ],
    "segfault": [("kernel", "worker[{pid}]: segfault at 0 ip 000055d0c0de1234 sp 00007ffd1234abcd error 4 in worker[55d0c0de0000+2000]")],
    "disk": [
        ("kernel", "blk_update_request: I/O error, dev sda, sector {p} op 0x0:(READ)"),
        ("kernel", "Buffer I/O error on dev sda1, logical block {n}, async page read"),
        ("kernel", "EXT4-fs error (device sda1): ext4_find_entry:1455: inode #{n}: comm ls: reading directory lblock 0"),
    ],
    "hung": [("kernel", "INFO: task kworker/u8:2:{pid} blocked for more than 120 seconds.")],
    "auth": [("sshd", "Failed password for invalid user admin from 203.0.113.{n} port {p} ssh2"),
             ("sshd", "Invalid user test from 203.0.113.{n} port {p}")],
    "service": [("systemd", "nginx.service: Main process exited, code=exited, status=1/FAILURE"),
                ("systemd", "nginx.service: Failed with result 'exit-code'."),
                ("systemd", "Failed to start A high performance web server and a reverse proxy server.")],
    "thermal": [("kernel", "CPU{n}: Core temperature above threshold, cpu clock throttled (total events = {p})")],
}


def sample_log(minutes: int = 180, seed: int = 42, host: str = "kernelsense-vm", end: datetime | None = None) -> str:
    """Generate a syslog-format log with normal traffic and a few incident bursts."""
    rng = random.Random(seed)
    end = (end or datetime.now()).replace(microsecond=0)
    start = end - timedelta(minutes=minutes)
    incident_minutes = {}
    kinds = list(_INCIDENTS)
    for k in rng.sample(kinds, k=min(5, len(kinds))):
        incident_minutes[rng.randrange(10, minutes - 5)] = k
    lines = []
    for m in range(minutes):
        t0 = start + timedelta(minutes=m)
        for _ in range(rng.randint(3, 9)):
            t = t0 + timedelta(seconds=rng.randrange(60))
            src, msg = rng.choice(_NORMAL)
            lines.append((t, src, msg))
        if m in incident_minutes:
            kind = incident_minutes[m]
            burst = 25 if kind == "auth" else rng.randint(4, 12)
            for _ in range(burst):
                t = t0 + timedelta(seconds=rng.randrange(60))
                src, msg = rng.choice(_INCIDENTS[kind])
                lines.append((t, src, msg))
    lines.sort(key=lambda x: x[0])
    out = []
    for t, src, msg in lines:
        pid = rng.randint(300, 9000)
        text = msg.format(n=rng.randint(1, 250), p=rng.randint(1000, 65000), pid=rng.randint(1000, 9000))
        tag = src if src == "kernel" else f"{src}[{pid}]"
        out.append(f"{t.strftime('%b')} {t.day:2d} {t.strftime('%H:%M:%S')} {host} {tag}: {text}")
    return "\n".join(out)
