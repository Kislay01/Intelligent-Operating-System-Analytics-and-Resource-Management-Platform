"""Memory leak simulation: allocates and keeps memory at a steady rate.

The resident set grows linearly, so the leak detector should flag it and the
memory forecast should show a rising trend. A hard cap prevents real harm.

    python workloads/memory_leak.py --rate 5 --max 800
"""

import argparse
import os
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=5.0, help="MB leaked per second")
    p.add_argument("--max", type=float, default=800.0, help="stop growing after this many MB")
    p.add_argument("--hold", type=float, default=60.0, help="seconds to hold memory after reaching max")
    a = p.parse_args()
    leaked = []
    total = 0.0
    print(f"memory_leak: PID {os.getpid()}, leaking {a.rate} MB/s up to {a.max} MB")
    try:
        while total < a.max:
            chunk = bytearray(int(a.rate * 2**20))
            for i in range(0, len(chunk), 4096):  # touch every page so it becomes resident
                chunk[i] = 1
            leaked.append(chunk)
            total += a.rate
            print(f"\r  resident leak: {total:6.0f} MB", end="", flush=True)
            time.sleep(1)
        print(f"\nreached {a.max} MB, holding for {a.hold:.0f}s")
        time.sleep(a.hold)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
