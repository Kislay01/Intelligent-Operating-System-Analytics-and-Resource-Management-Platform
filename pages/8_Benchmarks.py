import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.ui.common import INK, SERIES, concept, get_recorder, page_setup, show, sidebar_status, style

page_setup("Benchmarks", "📊")
sidebar_status(get_recorder())

concept("Measured results from <code>python benchmarks/run_all.py</code>: every number on this page is produced by a "
        "reproducible experiment and stored in <code>benchmarks/results/*.json</code>.")

ROOT = Path(__file__).resolve().parent.parent / "benchmarks"
results = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))}
if not results:
    st.info("No results yet. Run `python benchmarks/run_all.py`.")
    st.stop()

alloc = results.get("allocation", {})
if alloc.get("summary"):
    st.subheader("Closed-loop allocation: interactive wake-up latency under CPU contention")
    df = pd.DataFrame(alloc["summary"])
    c = st.columns(3)
    best = df.iloc[df["p99_ms"].idxmin()]
    q = df[df.policy == "Q-learning"]
    c[0].metric("Default p99 latency", f"{df.iloc[0]['p99_ms']:.2f} ms")
    if not q.empty:
        c[1].metric("Q-learning p99 latency", f"{q.iloc[0]['p99_ms']:.2f} ms",
                    f"-{q.iloc[0]['p99_reduction_%']:.0f}%", delta_color="inverse")
        c[2].metric("Batch throughput kept", f"{q.iloc[0]['batch_throughput_%']:.0f}%")
    fig = go.Figure(go.Bar(x=df["policy"], y=df["p99_ms"], error_y=dict(type="data", array=df["p99_std"].fillna(0)),
                           marker=dict(color=[SERIES[2] if p == "Q-learning" else SERIES[0] for p in df["policy"]],
                                       cornerradius=4),
                           text=df["p99_ms"].round(2), textposition="outside", textfont=dict(color=INK["secondary"]),
                           hovertemplate="%{x}: %{y:.2f} ms<extra></extra>"))
    fig.update_yaxes(title="p99 wake-up latency (ms)", rangemode="tozero")
    show(style(fig, 360, f"{alloc['batch_hogs']} CPU hogs on {alloc['cpus']} CPUs · {alloc['trials']} trials"), "alloc")
    st.dataframe(df, hide_index=True, width="stretch")

rmd = ROOT / "RESULTS.md"
if rmd.exists():
    with st.expander("Full results report", expanded=not alloc):
        st.markdown(rmd.read_text())
