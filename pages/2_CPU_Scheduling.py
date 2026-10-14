import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.scheduling import (
    FCFS, GOALS, MLFQ, SJF, SRTF, PriorityScheduler, RoundRobin, best_alpha, compare, exponential_average, recommend,
)
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

tab_sim, tab_cmp, tab_rec, tab_burst = st.tabs(["Simulate", "Compare all", "Recommendation", "Burst prediction"])

# ------------------------------------------------------------------ simulate
with tab_sim:
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ["FCFS", "SJF", "SRTF", "Round Robin", "Priority", "MLFQ"])
    q = c2.number_input("Time quantum", 1, 50, 4)
    pre = c3.checkbox("Preemptive priority", True) if algo == "Priority" else None
    aging = c3.number_input("Aging interval (0 = off)", 0, 100, 0) if algo == "Priority" else 0
    sched = {
        "FCFS": FCFS(), "SJF": SJF(), "SRTF": SRTF(), "Round Robin": RoundRobin(int(q)),
        "Priority": PriorityScheduler(bool(pre), int(aging) or None), "MLFQ": MLFQ((int(q), int(q) * 2)),
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
    results = compare(jobs, [FCFS(), SJF(), SRTF(), RoundRobin(int(qc)), PriorityScheduler(False),
                             PriorityScheduler(True, 10), MLFQ((int(qc), int(qc) * 2))])
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
