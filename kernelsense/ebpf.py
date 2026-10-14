"""Optional eBPF run-queue latency measurement.

Run-queue latency is the time between a task becoming runnable (sched_wakeup)
and actually getting a CPU (sched_switch). It is the most direct measure of
scheduler-induced delay. This module uses ``bpftrace`` when it is installed
(and the process has CAP_BPF/root); otherwise ``available()`` returns False and
the platform falls back to the user-space latency probe.

    sudo python -m kernelsense.ebpf 5      # 5-second histogram
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

RUNQLAT = r"""
tracepoint:sched:sched_wakeup, tracepoint:sched:sched_wakeup_new { @qtime[args->pid] = nsecs; }
tracepoint:sched:sched_switch {
  if (args->prev_state == 0) { @qtime[args->prev_pid] = nsecs; }
  $ns = @qtime[args->next_pid];
  if ($ns) { @usecs = hist((nsecs - $ns) / 1000); }
  delete(@qtime[args->next_pid]);
}
interval:s:%d { exit(); }
END { clear(@qtime); }
"""

_BUCKET = re.compile(r"^\[(\d+)(K|M)?,\s*(\d+)(K|M)?\)\s+(\d+)")
_SINGLE = re.compile(r"^\[(\d+)\]\s+(\d+)")
_SCALE = {None: 1, "K": 1024, "M": 1024 * 1024}


def available() -> bool:
    return bool(shutil.which("bpftrace")) and Path("/sys/kernel/btf/vmlinux").exists()


def parse_hist(text: str) -> list[dict]:
    """Parse a bpftrace power-of-two histogram into [{'lo_us','hi_us','count'}]."""
    rows = []
    for line in text.splitlines():
        single = _SINGLE.match(line.strip())
        if single:
            v = int(single.group(1))
            rows.append({"lo_us": v, "hi_us": v + 1, "count": int(single.group(2))})
            continue
        m = _BUCKET.match(line.strip())
        if m:
            lo = int(m.group(1)) * _SCALE[m.group(2)]
            hi = int(m.group(3)) * _SCALE[m.group(4)]
            rows.append({"lo_us": lo, "hi_us": hi, "count": int(m.group(5))})
    return rows


def percentile_from_hist(rows: list[dict], q: float) -> float | None:
    total = sum(r["count"] for r in rows)
    if not total:
        return None
    target = total * q / 100
    acc = 0
    for r in rows:
        acc += r["count"]
        if acc >= target:
            return float(r["hi_us"])
    return float(rows[-1]["hi_us"])


def runqlat(seconds: int = 5) -> dict:
    if not available():
        raise RuntimeError("bpftrace with BTF is required (sudo apt install bpftrace)")
    proc = subprocess.run(["bpftrace", "-e", RUNQLAT % seconds], capture_output=True, text=True,
                          timeout=seconds + 30, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "bpftrace failed (needs root/CAP_BPF)")
    hist = parse_hist(proc.stdout)
    return {"histogram": hist, "p50_us": percentile_from_hist(hist, 50), "p99_us": percentile_from_hist(hist, 99),
            "events": sum(r["count"] for r in hist)}


if __name__ == "__main__":
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(runqlat(secs))
