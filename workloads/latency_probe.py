"""Latency-sensitive probe: measures how long it waits for the CPU after waking.

Loop: sleep for ``--sleep-ms``, record how late it actually woke up
(wake-up latency = time the task sat runnable in the run queue plus timer
slack), then do ``--work-ms`` of CPU work (like handling a request). Under CPU
contention the wake-up latency grows; giving competing batch work a lower
priority shrinks it again. Results are written as JSON.

    python workloads/latency_probe.py --duration 10 --out probe.json
"""

import argparse
import json
import os
import time


def busy(ms: float) -> None:
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        pass


def percentile(values, q):
    v = sorted(values)
    if not v:
        return 0.0
    k = (len(v) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--sleep-ms", type=float, default=4.0)
    p.add_argument("--work-ms", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=1.0)
    p.add_argument("--out")
    a = p.parse_args()
    lat = []
    start = time.perf_counter()
    while (now := time.perf_counter()) - start < a.duration:
        t0 = time.perf_counter()
        time.sleep(a.sleep_ms / 1000)
        late_ms = (time.perf_counter() - t0) * 1000 - a.sleep_ms
        if now - start >= a.warmup:
            lat.append(late_ms)
        busy(a.work_ms)
    res = {
        "pid": os.getpid(),
        "samples": len(lat),
        "p50_ms": round(percentile(lat, 50), 3),
        "p95_ms": round(percentile(lat, 95), 3),
        "p99_ms": round(percentile(lat, 99), 3),
        "max_ms": round(max(lat) if lat else 0.0, 3),
        "mean_ms": round(sum(lat) / len(lat), 3) if lat else 0.0,
    }
    text = json.dumps(res)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
