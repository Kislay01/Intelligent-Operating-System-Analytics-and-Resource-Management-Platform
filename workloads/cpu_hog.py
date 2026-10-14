"""CPU-bound workload: busy-loop worker processes.

Each worker never blocks, so it accumulates involuntary context switches and
should be classified as CPU-bound.

    python workloads/cpu_hog.py --workers 2 --duration 120
"""

import argparse
import multiprocessing as mp
import os
import time


def burn(stop_at: float) -> None:
    x = 0
    while time.time() < stop_at:
        x = (x * 1103515245 + 12345) % 2**31


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--duration", type=float, default=120)
    a = p.parse_args()
    stop = time.time() + a.duration
    procs = [mp.Process(target=burn, args=(stop,), name=f"cpu_hog_{i}") for i in range(a.workers)]
    for pr in procs:
        pr.start()
    print(f"cpu_hog: {a.workers} workers, PIDs {[pr.pid for pr in procs]}, for {a.duration:.0f}s")
    try:
        for pr in procs:
            pr.join()
    except KeyboardInterrupt:
        for pr in procs:
            pr.terminate()


if __name__ == "__main__":
    main()
