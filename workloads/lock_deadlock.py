"""Real deadlock between two processes using file locks (flock).

Process A locks file1 then waits for file2; process B locks file2 then waits
for file1. Both block forever (flock has no deadlock detection), which shows up
in /proc/locks as two holders and two blocked waiters forming a cycle. The
Deadlock page's "Live system" tab detects it. Both processes exit after
--timeout seconds via SIGALRM so nothing is left hanging.

    python workloads/lock_deadlock.py --timeout 60
"""

import argparse
import fcntl
import multiprocessing as mp
import os
import signal
import tempfile
import time


def worker(first: str, second: str, name: str, timeout: int, barrier) -> None:
    signal.signal(signal.SIGALRM, lambda *_: os._exit(0))
    signal.alarm(timeout)
    with open(first, "w") as f1:
        fcntl.flock(f1, fcntl.LOCK_EX)
        print(f"{name} (PID {os.getpid()}) holds {os.path.basename(first)}, waiting for {os.path.basename(second)}")
        barrier.wait()
        with open(second, "w") as f2:
            fcntl.flock(f2, fcntl.LOCK_EX)  # blocks forever: the other process holds it
            print(f"{name} acquired both (no deadlock)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=60)
    a = p.parse_args()
    d = tempfile.mkdtemp(prefix="ks_locks_")
    f1, f2 = os.path.join(d, "resource1.lock"), os.path.join(d, "resource2.lock")
    barrier = mp.Barrier(2)
    pa = mp.Process(target=worker, args=(f1, f2, "A", a.timeout, barrier))
    pb = mp.Process(target=worker, args=(f2, f1, "B", a.timeout, barrier))
    pa.start()
    pb.start()
    print(f"Deadlock created for {a.timeout}s - open the Deadlock page > Live system, or `cat /proc/locks`.")
    pa.join()
    pb.join()
    time.sleep(0.1)
    print("Timeout reached, processes released.")


if __name__ == "__main__":
    main()
