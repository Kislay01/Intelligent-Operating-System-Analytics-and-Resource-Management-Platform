"""Create zombie processes: children exit but the parent never calls wait().

The children stay in state Z until the parent exits, which the process
monitor reports as a zombie anomaly.

    python workloads/zombie_maker.py --count 3 --duration 60
"""

import argparse
import os
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--duration", type=float, default=60)
    a = p.parse_args()
    kids = []
    for _ in range(a.count):
        pid = os.fork()
        if pid == 0:
            os._exit(0)  # child exits immediately
        kids.append(pid)
    print(f"zombie_maker: parent PID {os.getpid()}, zombie children {kids} for {a.duration:.0f}s")
    try:
        time.sleep(a.duration)  # never wait() -> children remain zombies
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
