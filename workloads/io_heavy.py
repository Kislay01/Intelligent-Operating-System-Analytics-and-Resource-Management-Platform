"""I/O-bound workload: repeatedly writes, syncs and reads a temporary file.

Blocking on disk produces many voluntary context switches with little CPU, so
the process should be classified as I/O-bound.

    python workloads/io_heavy.py --size 64 --duration 120
"""

import argparse
import os
import tempfile
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, help="file size in MB per cycle")
    p.add_argument("--duration", type=float, default=120)
    a = p.parse_args()
    block = os.urandom(1 << 20)
    stop = time.time() + a.duration
    print(f"io_heavy: PID {os.getpid()}, {a.size} MB write+fsync+read cycles for {a.duration:.0f}s")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "io_heavy.bin")
        cycles = 0
        try:
            while time.time() < stop:
                with open(path, "wb") as fh:
                    for _ in range(a.size):
                        fh.write(block)
                    fh.flush()
                    os.fsync(fh.fileno())
                with open(path, "rb") as fh:
                    while fh.read(1 << 20):
                        pass
                cycles += 1
        except KeyboardInterrupt:
            pass
    print(f"done, {cycles} cycles")


if __name__ == "__main__":
    main()
