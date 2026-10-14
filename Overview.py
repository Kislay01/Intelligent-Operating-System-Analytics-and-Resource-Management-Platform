"""KernelSense dashboard - entry point.

Run:  streamlit run Overview.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.monitoring import detect_anomalies, state_distribution
from kernelsense.monitoring.procfs import capabilities
from kernelsense.ui.common import (
    INK, SERIES, SEVERITY_ICON, concept, get_recorder, line_chart, page_setup, show, sidebar_status, style,
)

page_setup("KernelSense · System Overview", "🧠")
rec = get_recorder()
sidebar_status(rec)

with st.sidebar:
    refresh = st.slider("Refresh every (s)", 1, 30, 3)
    window = st.select_slider("History window", options=[5, 15, 30, 60, 180, 720], value=15,
                              format_func=lambda m: f"{m} min")

concept(
    "Intelligent Operating System Analytics and Resource Management Platform. Live data comes straight from the "
    "kernel (<code>/proc</code>, <code>/proc/pressure</code>); every module on the left applies a core OS concept "
    "- scheduling, memory management, deadlocks, paging, logging and resource allocation - to it."
)


@st.fragment(run_every=refresh)
def live():
    s, procs = rec.snapshot()
    if s is None:
        st.info("Collecting the first sample...")
        return

    c = st.columns(6)
    c[0].metric("CPU", f"{s.cpu_percent:.1f}%")
    c[1].metric(f"Memory ({s.mem_used_mb/1024:.1f} / {s.mem_total_mb/1024:.1f} GB)", f"{s.mem_percent:.1f}%")
    c[2].metric("Swap", f"{s.swap_percent:.1f}%")
    c[3].metric("Load avg (1 min)", f"{s.load1:.2f}", help=f"5 min: {s.load5:.2f} · 15 min: {s.load15:.2f}")
    c[4].metric(f"Processes ({s.runnable} runnable)", s.process_count)
    psi = (s.pressure or {}).get("cpu") or {}
    c[5].metric("CPU pressure (PSI)", f"{psi.get('some', {}).get('avg10', 0):.2f}%" if psi else "n/a",
                help="Share of time at least one task was waiting for CPU (last 10 s). Needs PSI support.")

    hist = rec.store.system_history(window)
    left, right = st.columns(2)
    with left:
        if not hist.empty:
            show(line_chart(hist, "time", {"cpu_percent": "CPU %"}, "CPU utilisation", "%", 260, [0, 100]), "cpu")
            show(line_chart(hist, "time", {"load1": "1 min", "load5": "5 min", "load15": "15 min"},
                            "Load average (runnable + uninterruptible tasks)", "", 260), "load")
    with right:
        if not hist.empty:
            show(line_chart(hist, "time", {"mem_percent": "Memory %"}, "Memory utilisation", "%", 260, [0, 100]), "mem")
        cores = pd.DataFrame({"core": [f"CPU{i}" for i in range(len(s.per_cpu))], "util": s.per_cpu})
        fig = go.Figure(go.Bar(x=cores["core"], y=cores["util"], marker=dict(color=SERIES[0], cornerradius=4),
                               text=[f"{v:.0f}%" for v in cores["util"]], textposition="outside",
                               textfont=dict(color=INK["secondary"]),
                               hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(range=[0, 110], title="%")
        show(style(fig, 260, "Per-core utilisation"), "cores")

    a, b = st.columns([1, 1])
    with a:
        st.subheader("Process states")
        dist = state_distribution(procs)
        st.dataframe(pd.DataFrame({"state": list(dist), "count": list(dist.values())}),
                     hide_index=True, width="stretch")
        if s.pressure:
            rows = []
            for res, val in s.pressure.items():
                if val:
                    rows.append({"resource": res, "some avg10 %": val.get("some", {}).get("avg10"),
                                 "full avg10 %": val.get("full", {}).get("avg10")})
            if rows:
                st.caption("Pressure Stall Information - time tasks spent stalled waiting for each resource")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with b:
        st.subheader("Top processes")
        top = sorted(procs, key=lambda p: (p.cpu_percent, p.rss_mb), reverse=True)[:10]
        st.dataframe(
            pd.DataFrame([{"PID": p.pid, "Name": p.name, "State": p.state, "CPU %": p.cpu_percent,
                           "RSS MB": p.rss_mb, "Class": p.workload_class} for p in top]),
            hide_index=True, width="stretch",
        )
        anomalies = detect_anomalies(procs)
        st.subheader(f"Alerts ({len(anomalies)})")
        for an in anomalies[:6]:
            st.write(f"{SEVERITY_ICON.get(an.severity, '•')} **{an.kind}** - {an.name} ({an.pid}): {an.detail}")
        if not anomalies:
            st.write(f"{SEVERITY_ICON['good']} No process anomalies detected.")


live()

with st.expander("Kernel interfaces available on this machine"):
    caps = capabilities()
    st.dataframe(pd.DataFrame({"interface": list(caps), "available": ["✅" if v else "❌" for v in caps.values()]}),
                 hide_index=True)
    st.caption("Modules adapt automatically: unavailable interfaces are skipped instead of failing.")
