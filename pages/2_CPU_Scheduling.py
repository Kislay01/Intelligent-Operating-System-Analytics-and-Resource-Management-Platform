import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.scheduling import (
    CFS, EEVDF, FCFS, GOALS, MLFQ, SJF, SRTF, Lottery, PriorityScheduler, RoundRobin, Stride, best_alpha, compare,
    cpu_share, default_schedulers, exponential_average, nice_weight, recommend,
)
from kernelsense.scheduling.models import Job
from kernelsense.scheduling.workload import PROFILES, from_frame, from_live_processes, synthetic, to_frame
from kernelsense.ui.common import INK, SERIES, concept, df_download, gantt, get_recorder, page_setup, show, sidebar_status, style

page_setup("CPU Scheduling Recommendation", "⏱️")
rec = get_recorder()
sidebar_status(rec)

concept(
    "The short-term scheduler decides which ready process gets the CPU next. Policies trade off "
    "<b>waiting time</b>, <b>turnaround time</b>, <b>response time</b>, <b>throughput</b> and <b>fairness</b>, "
    "and preemptive policies pay for responsiveness with <b>context switches</b>. Waiting = Turnaround − Burst; "
    "Turnaround = Completion − Arrival; Response = First run − Arrival."
)

# ------------------------------------------------------------------ workload
st.subheader("1 · Workload")
src = st.radio("Source", ["Synthetic profile", "Edit table", "Live processes", "Upload CSV"], horizontal=True)
if "sched_jobs" not in st.session_state:
    st.session_state.sched_jobs = to_frame(synthetic("balanced"))

if src == "Synthetic profile":
    c1, c2, c3 = st.columns([2, 1, 1])
    profile = c1.selectbox("Profile", list(PROFILES), format_func=lambda p: f"{p} - {PROFILES[p]}")
    n = c2.number_input("Processes", 3, 20, 8)
    seed = c3.number_input("Seed", 0, 9999, 42)
    st.session_state.sched_jobs = to_frame(synthetic(profile, int(n), int(seed)))
elif src == "Live processes":
    _, procs = rec.snapshot()
    jobs = from_live_processes(procs, limit=10)
    if jobs:
        st.session_state.sched_jobs = to_frame(jobs)
        st.caption("Bursts are estimated from each process's current CPU demand; priority = nice + 20. "
                   "The OS never knows future bursts, so this is an approximation of the live mix.")
    else:
        st.warning("No active processes right now; keeping the previous workload.")
elif src == "Upload CSV":
    up = st.file_uploader("CSV with columns pid, arrival, burst[, priority]", type="csv")
    if up is not None:
        try:
            st.session_state.sched_jobs = to_frame(from_frame(pd.read_csv(up)))
        except ValueError as e:
            st.error(str(e))

edited = st.data_editor(st.session_state.sched_jobs, num_rows="dynamic", width="stretch",
                        disabled=src != "Edit table", key=f"editor_{src}")
try:
    jobs = from_frame(edited.dropna(subset=["pid", "arrival", "burst"]))
except (ValueError, TypeError) as e:
    st.error(f"Invalid workload: {e}")
    st.stop()
if not jobs:
    st.stop()

tab_sim, tab_cmp, tab_rec, tab_burst, tab_prop, tab_trace, tab_ml = st.tabs(
    ["Simulate", "Compare all", "Recommendation", "Burst prediction", "Proportional share (CFS/EEVDF)",
     "Live trace replay", "ML burst prediction"])

# ------------------------------------------------------------------ simulate
with tab_sim:
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["FCFS", "SJF", "SRTF", "Round Robin", "Priority", "MLFQ", "CFS", "EEVDF",
                                      "Stride", "Lottery"])
    q = c2.number_input("Time quantum", 1, 50, 4)
    pre = c3.checkbox("Preemptive priority", True) if algo == "Priority" else None
    aging = c3.number_input("Aging interval (0 = off)", 0, 100, 0) if algo == "Priority" else 0
    sched = {
        "FCFS": FCFS(), "SJF": SJF(), "SRTF": SRTF(), "Round Robin": RoundRobin(int(q)),
        "Priority": PriorityScheduler(bool(pre), int(aging) or None), "MLFQ": MLFQ((int(q), int(q) * 2)),
        "CFS": CFS(sched_latency=max(2, int(q) * 2)), "EEVDF": EEVDF(base_slice=int(q)), "Stride": Stride(),
        "Lottery": Lottery(),
    }[algo]
    res = sched.run(jobs)
    show(gantt(res.gantt_frame(), f"Gantt chart - {res.algorithm}"), "gantt_one")
    summ = res.summary()
    c = st.columns(6)
    c[0].metric("Avg waiting", summ["avg_waiting"])
    c[1].metric("Avg turnaround", summ["avg_turnaround"])
    c[2].metric("Avg response", summ["avg_response"])
    c[3].metric("Throughput", f"{summ['throughput']:.3f}/unit")
    c[4].metric("Context switches", summ["context_switches"])
    c[5].metric("Fairness (Jain)", summ["fairness"])
    st.dataframe(res.stats_frame(), hide_index=True, width="stretch")
    if algo == "MLFQ":
        g = res.gantt_frame()
        st.caption("Queue level of each slice (0 = highest priority). Jobs that use their whole quantum are demoted.")
        st.dataframe(g[g["pid"] != "IDLE"][["pid", "start", "end", "queue"]].T, width="stretch")

# ------------------------------------------------------------------ compare
with tab_cmp:
    qc = st.number_input("Quantum for RR / MLFQ", 1, 50, 4, key="cmp_q")
    results = compare(jobs, default_schedulers(int(qc)))
    table = pd.DataFrame([r.summary() for r in results])
    st.dataframe(table, hide_index=True, width="stretch")
    df_download(table, "scheduling_comparison.csv")
    cols = st.columns(3)
    for col, (metric, label) in zip(cols, [("avg_waiting", "Average waiting time"),
                                           ("avg_turnaround", "Average turnaround time"),
                                           ("avg_response", "Average response time")]):
        best = table[metric].min()
        colors = [SERIES[2] if v == best else SERIES[0] for v in table[metric]]
        fig = go.Figure(go.Bar(x=table[metric], y=table["algorithm"], orientation="h",
                               marker=dict(color=colors, cornerradius=4), text=table[metric],
                               textposition="outside", textfont=dict(color=INK["secondary"]),
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        with col:
            show(style(fig, 330, label + " (lowest highlighted)"), f"cmp_{metric}")
    with st.expander("Gantt chart of every algorithm"):
        for r in results:
            show(gantt(r.gantt_frame(), r.algorithm, 150), f"g_{r.algorithm}")

# ------------------------------------------------------------------ recommend
with tab_rec:
    c1, c2, c3 = st.columns([2, 1, 1])
    goal = c1.selectbox("Optimisation goal", list(GOALS), format_func=lambda g: f"{g} - {GOALS[g]}")
    practical = c2.checkbox("Practical only", value=False,
                            help="Exclude SJF/SRTF, which need to know future CPU bursts.")
    rq = c3.number_input("Quantum (0 = auto)", 0, 50, 0)
    r = recommend(jobs, goal, int(rq) or None, practical)
    st.success(f"**Recommended: {r.best}** · suggested time quantum = {r.suggested_quantum}")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Why**")
        for reason in r.reasons:
            st.markdown(f"- {reason}")
        st.markdown(f"**On Linux:** {r.linux_hint}")
    with b:
        p = r.profile.as_dict()
        st.markdown("**Workload profile**")
        st.dataframe(pd.DataFrame({"feature": list(p), "value": [str(v) for v in p.values()]}),
                     hide_index=True, width="stretch")
    st.markdown("**Simulation-validated ranking** (score = weighted, normalised metrics for this goal; lower is better)")
    st.dataframe(r.ranking[["rank", "algorithm", "score", "avg_waiting", "avg_turnaround", "avg_response",
                            "context_switches", "fairness"]], hide_index=True, width="stretch")
    show(gantt(r.results[r.best].gantt_frame(), f"Recommended schedule - {r.best}"), "gantt_best")

# ------------------------------------------------------------------ burst
with tab_burst:
    st.markdown("Exponential averaging predicts the next CPU burst: "
                "**τ<sub>n+1</sub> = α·t<sub>n</sub> + (1 − α)·τ<sub>n</sub>**", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 1, 1])
    text = c1.text_input("Observed CPU bursts", "6 4 6 4 13 13 13 4 5 6")
    alpha = c2.slider("α", 0.0, 1.0, 0.5, 0.05)
    tau0 = c3.number_input("τ₀ (initial guess)", 0.0, 100.0, 10.0)
    try:
        bursts = [float(x) for x in text.replace(",", " ").split()]
    except ValueError:
        st.error("Bursts must be numbers")
        st.stop()
    if bursts:
        trace = exponential_average(bursts, alpha, tau0)
        idx = list(range(1, len(bursts) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=trace.actual, name="Actual burst tₙ", mode="lines+markers",
                                 line=dict(color=SERIES[0], width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=idx + [len(idx) + 1], y=trace.predicted + [trace.next_prediction],
                                 name="Predicted τₙ", mode="lines+markers",
                                 line=dict(color=SERIES[1], width=2, dash="dash"), marker=dict(size=8)))
        fig.update_layout(hovermode="x unified", xaxis_title="burst #", yaxis_title="time units")
        show(style(fig, 320, "Actual vs predicted CPU bursts"), "burst")
        a_best, mae_best = best_alpha(bursts)
        c = st.columns(3)
        c[0].metric("Next burst prediction", f"{trace.next_prediction:.2f}")
        c[1].metric("MAE (this α)", f"{trace.mae:.2f}")
        c[2].metric(f"Best α on this history (MAE {mae_best:.2f})", f"{a_best:.2f}")
        st.caption("α = 0 ignores recent behaviour; α = 1 trusts only the last burst. SJF/SRTF in a real kernel "
                   "would order processes by this prediction.")

# ------------------------------------------------------------ proportional
with tab_prop:
    st.markdown("CFS, EEVDF, Stride and Lottery share the CPU in proportion to **weights** derived from nice values "
                "(kernel table: nice 0 = 1024, each step ≈ ×1.25). Three always-runnable processes:")
    c1, c2, c3 = st.columns(3)
    nices = [c1.slider("Nice of A", 0, 19, 0), c2.slider("Nice of B", 0, 19, 5), c3.slider("Nice of C", 0, 19, 10)]
    pj = [Job(n, 0, 400, nv) for n, nv in zip("ABC", nices)]
    base = min(nices)
    weights = {n: nice_weight(nv - base) for n, nv in zip("ABC", nices)}
    total = sum(weights.values())
    rows = []
    for sch in (CFS(), EEVDF(), Stride(), Lottery()):
        share = cpu_share(sch.run(pj), 0, 300)
        for n in "ABC":
            rows.append({"policy": sch.name, "process": n, "share": share.get(n, 0)})
    sdf = pd.DataFrame(rows)
    fig = go.Figure()
    for i, n in enumerate("ABC"):
        d = sdf[sdf["process"] == n]
        fig.add_trace(go.Bar(x=d["policy"], y=d["share"], name=f"{n} (nice {nices[i]})",
                             marker=dict(color=SERIES[i], cornerradius=4),
                             hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.add_hline(y=weights[n] / total, line=dict(color=SERIES[i], dash="dot", width=1))
    fig.update_layout(barmode="group", yaxis=dict(tickformat=".0%", title="CPU share (first 300 ticks)"))
    show(style(fig, 360, "Measured CPU share vs weight-proportional target (dotted)"), "propshare")
    st.caption("CFS picks the smallest virtual runtime (vruntime += Δt × 1024 / weight); EEVDF picks the eligible "
               "task with the earliest virtual deadline; Stride is the deterministic and Lottery the randomised "
               "proportional-share scheduler.")

# ------------------------------------------------------------- live trace
with tab_trace:
    from kernelsense.scheduling.trace import capture

    st.markdown("Sample every process's CPU-time counter (`utime + stime` in `/proc/[pid]/stat`) and cut it into "
                "**CPU bursts**. Each observed burst is replayed as a job, so the schedulers are compared on what "
                "this machine actually ran.")
    c1, c2, c3 = st.columns(3)
    dur = c1.slider("Capture seconds", 1, 15, 4)
    ivl = c2.select_slider("Sampling interval (ms)", [10, 20, 50, 100], 20)
    if c3.button("Capture trace", type="primary", width="stretch"):
        with st.spinner("Capturing CPU bursts..."):
            st.session_state.trace = capture(dur, ivl)
    tr = st.session_state.get("trace")
    if tr is None:
        st.info("Capture a trace (start `python workloads/cpu_hog.py` or `io_heavy.py` for a more interesting mix).")
    elif not tr.bursts:
        st.warning("No CPU activity captured.")
    else:
        df = tr.frame()
        c = st.columns(3)
        c[0].metric("Bursts", len(df))
        c[1].metric("Processes", df["pid"].nunique())
        c[2].metric("Median burst", f"{df['length_ms'].median():.0f} ms")
        top = df.groupby("name")["length_ms"].agg(["count", "mean", "max"]).sort_values("count", ascending=False)
        st.dataframe(top.round(1).head(12), width="stretch")
        tjobs = tr.to_jobs()
        if len(tjobs) >= 3:
            rr = recommend(tjobs, "interactive", practical_only=True)
            st.success(f"Replayed {len(tjobs)} bursts · best practical policy for responsiveness: **{rr.best}**")
            st.dataframe(rr.ranking[["rank", "algorithm", "avg_waiting", "avg_response", "p95_response",
                                     "context_switches"]], hide_index=True, width="stretch")
            show(gantt(rr.results[rr.best].gantt_frame(), f"Replayed trace - {rr.best}", 190), "tracegantt")

# ---------------------------------------------------------------- ML burst
with tab_ml:
    from kernelsense.scheduling.ml_burst import compare_predictors, summary, synthetic_sequences

    st.markdown("A gradient-boosting regressor over the last bursts, their trend and the exponential average itself, "
                "versus exponential averaging. Evaluated walk-forward on **held-out** sequences.")

    @st.cache_data(show_spinner="Training and evaluating predictors...")
    def burst_eval(seed: int):
        seqs = synthetic_sequences(seed=seed)
        return compare_predictors(seqs[:90], seqs[90:])

    table = burst_eval(0)
    summ = summary(table)
    st.dataframe(summ, hide_index=True, width="stretch")
    fig = go.Figure(go.Bar(x=summ["MAE"], y=summ["method"], orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                           text=summ["MAE"], textposition="outside", textfont=dict(color=INK["secondary"])))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, 260, "Mean absolute error of next-burst prediction (lower is better)"), "mlburst")
    tr = st.session_state.get("trace")
    if tr is not None and tr.sequences():
        real = list(tr.sequences().values())
        st.markdown(f"**On your captured trace** ({len(real)} processes with ≥3 bursts):")
        st.dataframe(summary(compare_predictors(synthetic_sequences(seed=0), real)), hide_index=True, width="stretch")
