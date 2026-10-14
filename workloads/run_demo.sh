#!/usr/bin/env bash
# Start every demo workload in the background for a live demonstration.
# Usage: bash workloads/run_demo.sh [duration_seconds]
set -euo pipefail
DUR="${1:-180}"
DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$DIR/cpu_hog.py" --workers 1 --duration "$DUR" &
python3 "$DIR/memory_leak.py" --rate 4 --max 600 --hold "$DUR" &
python3 "$DIR/io_heavy.py" --size 32 --duration "$DUR" &
python3 "$DIR/zombie_maker.py" --count 2 --duration "$DUR" &
python3 "$DIR/lock_deadlock.py" --timeout "$DUR" &

echo "Demo workloads running for ${DUR}s. Press Ctrl+C to stop them all."
trap 'kill $(jobs -p) 2>/dev/null; exit 0' INT TERM
wait
