"""Command-line interface.

Examples:
    python -m kernelsense status
    python -m kernelsense record --interval 2 --duration 600
    python -m kernelsense schedule --profile convoy --goal interactive
    python -m kernelsense paging --refs "7 0 1 2 0 3 0 4 2 3 0 3 2" --frames 3
    python -m kernelsense bankers
    python -m kernelsense logs --source auto
    python -m kernelsense allocate --apply
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def cmd_status(args) -> int:
    from .monitoring import PsutilCollector, detect_anomalies, state_distribution
    from .monitoring.procfs import capabilities

    c = PsutilCollector()
    c.processes()
    time.sleep(1)
    s, procs = c.system(), c.processes()
    print("Kernel capabilities:", capabilities())
    print(f"CPU {s.cpu_percent:.1f}% | Mem {s.mem_percent:.1f}% | Swap {s.swap_percent:.1f}% | "
          f"Load {s.load1:.2f} {s.load5:.2f} {s.load15:.2f} | Processes {s.process_count}")
    if s.pressure.get("cpu"):
        print("PSI cpu some avg10:", s.pressure["cpu"].get("some", {}).get("avg10"))
    print("States:", state_distribution(procs))
    top = sorted(procs, key=lambda p: p.cpu_percent, reverse=True)[:10]
    print(pd.DataFrame([{"pid": p.pid, "name": p.name, "state": p.state, "cpu%": p.cpu_percent,
                         "rss_mb": p.rss_mb, "class": p.workload_class} for p in top]).to_string(index=False))
    for a in detect_anomalies(procs)[:10]:
        print(f"[{a.severity}] {a.kind}: {a.name} ({a.pid}) - {a.detail}")
    return 0


def cmd_record(args) -> int:
    from .recorder import Recorder

    rec = Recorder(interval=args.interval)
    print(f"Recording every {rec.interval}s to {rec.store.db_path} (Ctrl+C to stop)")
    end = time.time() + args.duration if args.duration else None
    try:
        while end is None or time.time() < end:
            rec.sample_once()
            s = rec.latest_system
            print(f"\r#{rec.samples} cpu={s.cpu_percent:5.1f}% mem={s.mem_percent:5.1f}% "
                  f"overhead={rec.overhead_ms:6.1f}ms", end="", flush=True)
            time.sleep(rec.interval)
    except KeyboardInterrupt:
        pass
    print(f"\nStored {rec.store.row_count()} system samples.")
    return 0


def cmd_schedule(args) -> int:
    from .scheduling import recommend
    from .scheduling.workload import from_csv, synthetic

    jobs = from_csv(args.csv) if args.csv else synthetic(args.profile, args.n, args.seed)
    rec = recommend(jobs, args.goal, args.quantum, practical_only=args.practical)
    print(pd.DataFrame([j.__dict__ for j in jobs]).to_string(index=False))
    print("\n" + rec.ranking[["rank", "algorithm", "avg_waiting", "avg_turnaround", "avg_response",
                              "context_switches", "fairness", "score"]].to_string(index=False))
    print(f"\nRecommended for '{args.goal}': {rec.best}  (suggested quantum {rec.suggested_quantum})")
    for r in rec.reasons:
        print(" -", r)
    print("Linux:", rec.linux_hint)
    return 0


def cmd_paging(args) -> int:
    from .paging import compare, detect_belady, generate, parse_refs

    refs = parse_refs(args.refs) if args.refs else generate(args.pattern, args.length, args.pages)
    print("Reference string:", " ".join(map(str, refs)))
    print(pd.DataFrame([r.summary() for r in compare(refs, args.frames)]).to_string(index=False))
    for case in detect_belady(refs, max_frames=max(args.frames + 3, 6)):
        print(f"Belady's anomaly (FIFO): {case.frames_low} frames -> {case.faults_low} faults, "
              f"{case.frames_high} frames -> {case.faults_high} faults")
    return 0


def cmd_bankers(args) -> int:
    from .deadlock import bankers_risk, textbook_example

    s = textbook_example()
    res = s.safety()
    print("Need matrix:", s.need)
    print("Safe" if res.safe else "UNSAFE", "sequence:", res.sequence)
    risk = bankers_risk(s)
    print(f"Risk {risk.score} ({risk.level}):", "; ".join(risk.notes))
    if args.request:
        proc, *vals = args.request.split()
        r = s.request(proc, [int(v) for v in vals])
        print(r.reason)
    return 0


def cmd_logs(args) -> int:
    from . import logs

    name, df = logs.load(args.source, path=args.path)
    print(f"Source: {name}, {len(df)} lines")
    if df.empty:
        return 1
    print(df["severity"].value_counts().to_string())
    print(logs.event_summary(logs.detect_events(df)).to_string(index=False))
    an = logs.detect_anomalies(df)
    if not an.empty:
        print("Anomalous windows:")
        print(an[an["anomaly"]][["lines", "errors", "os_events", "error_z", "iforest_score"]].to_string())
    return 0


def cmd_allocate(args) -> int:
    from .allocator import Actuator, SystemState, from_snapshot, recommend
    from .monitoring import PsutilCollector

    c = PsutilCollector()
    c.processes()
    time.sleep(args.window)
    s, procs = c.system(), c.processes()
    active = [p for p in procs if p.cpu_percent > 0.5 or p.rss_mb > 200 or p.io_rate > 0]
    classified, actions = recommend(from_snapshot(active), SystemState(s.cpu_percent, s.mem_percent))
    print(classified[["pid", "name", "cpu_mean", "rss_mb", "predicted", "confidence"]]
          .sort_values("cpu_mean", ascending=False).head(15).to_string(index=False))
    act = Actuator(dry_run=not args.apply)
    for r in act.execute_all(actions):
        print(f"[{r.status}] {r.name} ({r.pid}) {r.kind}: {r.message}")
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="kernelsense", description="Intelligent OS analytics and resource management")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="system snapshot, top processes and anomalies").set_defaults(func=cmd_status)

    r = sub.add_parser("record", help="record metrics to the database")
    r.add_argument("--interval", type=float, default=None)
    r.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl+C)")
    r.set_defaults(func=cmd_record)

    s = sub.add_parser("schedule", help="compare schedulers and recommend one")
    s.add_argument("--profile", default="balanced")
    s.add_argument("--csv", help="CSV with pid,arrival,burst[,priority]")
    s.add_argument("--goal", default="balanced", choices=["balanced", "interactive", "throughput", "fairness"])
    s.add_argument("--quantum", type=int)
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--practical", action="store_true", help="exclude SJF/SRTF (need future knowledge)")
    s.set_defaults(func=cmd_schedule)

    g = sub.add_parser("paging", help="page replacement comparison")
    g.add_argument("--refs")
    g.add_argument("--frames", type=int, default=3)
    g.add_argument("--pattern", default="locality")
    g.add_argument("--length", type=int, default=30)
    g.add_argument("--pages", type=int, default=10)
    g.set_defaults(func=cmd_paging)

    b = sub.add_parser("bankers", help="Banker's algorithm on the textbook example")
    b.add_argument("--request", help='e.g. "P1 1 0 2"')
    b.set_defaults(func=cmd_bankers)

    lg = sub.add_parser("logs", help="log analysis")
    lg.add_argument("--source", default="auto", choices=["auto", "journald", "syslog", "dmesg", "sample"])
    lg.add_argument("--path")
    lg.set_defaults(func=cmd_logs)

    a = sub.add_parser("allocate", help="classify workloads and plan resource actions")
    a.add_argument("--apply", action="store_true", help="apply actions (default is dry-run)")
    a.add_argument("--window", type=float, default=2.0)
    a.set_defaults(func=cmd_allocate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
