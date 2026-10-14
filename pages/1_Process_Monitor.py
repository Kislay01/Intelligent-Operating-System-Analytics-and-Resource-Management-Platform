import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.monitoring import detect_anomalies, render_tree, to_frame
from kernelsense.monitoring.analysis import tree_edges
from kernelsense.monitoring.classify import CLASSES
from kernelsense.monitoring.procfs import STATE_NAMES, read_status
from kernelsense.ui.common import (
    INK, SERIES, SEVERITY_ICON, concept, get_recorder, line_chart, page_setup, show, sidebar_status, style,
)

page_setup("Process Monitoring", "🔍")
rec = get_recorder()
sidebar_status(rec)

concept(
    "Each process has a PCB the kernel exposes under <code>/proc/[pid]</code>: state (R running, S sleeping, "
    "D uninterruptible, Z zombie, T stopped), CPU time, resident memory, threads and context switches. "
    "<b>Voluntary</b> switches mean the process blocked (I/O, sleep); <b>involuntary</b> switches mean the "
    "scheduler preempted it - this is how processes are classified as CPU-bound or I/O-bound."
)

s, procs = rec.snapshot()
if not procs:
    st.info("Waiting for the first sample...")
    st.stop()

df = to_frame(procs)

with st.sidebar:
    st.subheader("Filters")
    states = st.multiselect("State", sorted(df["state"].unique()), format_func=lambda x: f"{x} - {STATE_NAMES.get(x, x)}")
    classes = st.multiselect("Workload class", [c for c in CLASSES + ["Unknown"] if c in set(df["workload_class"])])
    users = st.multiselect("User", sorted(df["username"].unique()))
    query = st.text_input("Name contains")
    hide_kernel = st.checkbox("Hide kernel threads", value=True)

view = df.copy()
if states:
    view = view[view["state"].isin(states)]
if classes:
    view = view[view["workload_class"].isin(classes)]
if users:
    view = view[view["username"].isin(users)]
if query:
    view = view[view["name"].str.contains(query, case=False, na=False)]
if hide_kernel:
    view = view[~((view["ppid"] == 2) | (view["pid"] == 2))]

tab_table, tab_tree, tab_detail, tab_alerts = st.tabs(["Process table", "Process tree", "Process detail", "Anomalies"])

with tab_table:
    c = st.columns(4)
    c[0].metric("Shown", len(view))
    c[1].metric("Threads (shown)", int(view["num_threads"].sum()))
    c[2].metric("Zombies", int((df["state"] == "Z").sum()))
    c[3].metric("In D state", int((df["state"] == "D").sum()))
    cols = ["pid", "ppid", "name", "username", "state", "cpu_percent", "rss_mb", "mem_percent", "num_threads",
            "nice", "vol_ctx_rate", "invol_ctx_rate", "minflt", "majflt", "num_fds", "workload_class"]
    st.dataframe(
        view[cols].sort_values(["cpu_percent", "rss_mb"], ascending=False),
        hide_index=True, width="stretch", height=420,
        column_config={
            "cpu_percent": st.column_config.ProgressColumn("CPU %", min_value=0, max_value=100, format="%.1f"),
            "rss_mb": st.column_config.NumberColumn("RSS MB", format="%.1f"),
            "vol_ctx_rate": st.column_config.NumberColumn("vol cs/s", help="voluntary context switches per second"),
            "invol_ctx_rate": st.column_config.NumberColumn("invol cs/s", help="involuntary context switches per second"),
            "minflt": "minor faults", "majflt": "major faults", "num_fds": "FDs",
        },
    )
    counts = df["workload_class"].value_counts()
    counts = counts.reindex([c for c in CLASSES + ["Unknown"] if c in counts.index])
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                           marker=dict(color=SERIES[0], cornerradius=4), text=counts.values,
                           textposition="outside", textfont=dict(color=INK["secondary"]),
                           hovertemplate="%{y}: %{x} processes<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, 240, "Workload classification of all processes"))

with tab_tree:
    left, right = st.columns([1, 1])
    with left:
        st.caption("Parent → child hierarchy (fork/exec), like `pstree -p`")
        st.code(render_tree([p for p in procs if not (hide_kernel and (p.ppid == 2 or p.pid == 2))]) or "(empty)",
                language=None)
    with right:
        edges = tree_edges([p for p in procs if not (p.ppid == 2 or p.pid == 2)])
        fig = go.Figure(go.Sunburst(ids=edges["id"], parents=edges["parent"], labels=edges["label"],
                                    values=edges["rss_mb"], branchvalues="remainder",
                                    marker=dict(colors=edges["rss_mb"], colorscale=[[0, "#cde2fb"], [1, "#184f95"]]),
                                    hovertemplate="%{label}<br>RSS %{value:.1f} MB<extra></extra>"))
        show(style(fig, 520, "Process tree sized by resident memory"))

with tab_detail:
    options = view.sort_values("cpu_percent", ascending=False)
    pid = st.selectbox("Process", options["pid"], format_func=lambda p: f"{p} - {df.set_index('pid').loc[p, 'name']}")
    if pid is not None:
        row = df.set_index("pid").loc[pid]
        c = st.columns(5)
        c[0].metric("State", f"{row['state']} ({row['state_name'].split(' ')[0]})")
        c[1].metric("CPU", f"{row['cpu_percent']:.1f}%")
        c[2].metric("RSS", f"{row['rss_mb']:.1f} MB")
        c[3].metric("Threads", int(row["num_threads"]))
        c[4].metric("Class", row["workload_class"])
        st.caption(f"Command: `{row['cmdline'] or row['name']}`")
        hist = rec.store.process_history(int(pid), minutes=60)
        if len(hist) >= 2:
            a, b = st.columns(2)
            with a:
                show(line_chart(hist, "time", {"cpu_percent": "CPU %"}, "CPU % history", "%", 260), "pcpu")
            with b:
                show(line_chart(hist, "time", {"rss_mb": "RSS MB"}, "Resident memory history", "MB", 260), "prss")
        else:
            st.info("History appears once the recorder has stored a few samples for this process "
                    "(only the busiest processes are stored).")
        status = read_status(int(pid))
        if status:
            keys = ["State", "PPid", "Threads", "VmPeak", "VmSize", "VmRSS", "VmSwap", "voluntary_ctxt_switches",
                    "nonvoluntary_ctxt_switches", "Cpus_allowed_list"]
            st.caption("Raw kernel view: /proc/[pid]/status")
            st.dataframe(pd.DataFrame({"field": [k for k in keys if k in status],
                                       "value": [status[k] for k in keys if k in status]}),
                         hide_index=True, width="stretch")

with tab_alerts:
    anomalies = detect_anomalies(procs)
    if not anomalies:
        st.success("No anomalies: no zombies, hung tasks, fork storms or descriptor leaks detected.")
    for a in anomalies:
        st.write(f"{SEVERITY_ICON.get(a.severity, '•')} **{a.kind}** · {a.name} (PID {a.pid}) - {a.detail}")
    with st.expander("What each alert means"):
        st.markdown(
            "- **Zombie**: process exited but the parent has not called `wait()`; it keeps a PID and PCB entry.\n"
            "- **Uninterruptible sleep (D)**: blocked in the kernel, cannot be killed; long D-state suggests stuck I/O.\n"
            "- **Orphan**: parent exited, so init (PID 1) adopted it.\n"
            "- **Fork storm**: a parent with an unusually large number of children (fork bomb pattern).\n"
            "- **FD leak / thread explosion**: resources growing without bound."
        )
