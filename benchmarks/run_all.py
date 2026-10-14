"""Run every experiment and regenerate benchmarks/RESULTS.md.

    python benchmarks/run_all.py                    # everything (~5-8 minutes)
    python benchmarks/run_all.py paging loghub      # selected experiments
    python benchmarks/run_all.py --report-only      # re-render RESULTS.md from saved JSON

The allocation experiment changes nice values of processes it starts itself and,
when run as root, uses a temporary cgroup under /sys/fs/cgroup/.../kernelsense.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from kernelsense.bench.experiments import EXPERIMENTS, run  # noqa: E402
from kernelsense.bench.report import render  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = [a for a in args if a not in EXPERIMENTS]
    if unknown:
        sys.exit(f"unknown experiments {unknown}; choose from {list(EXPERIMENTS)}")
    results_dir = ROOT / "results"
    if "--report-only" not in sys.argv:
        run(args or None, results_dir)
    print("wrote", render(results_dir, ROOT / "RESULTS.md"))


if __name__ == "__main__":
    main()
