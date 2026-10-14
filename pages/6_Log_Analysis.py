import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense import logs
from kernelsense.logs.sources import available_sources
from kernelsense.ui.common import INK, SERIES, STATUS, concept, get_recorder, page_setup, show, sidebar_status, style

page_setup("Log Analysis", "📜")
sidebar_status(get_recorder())

concept(
    "The kernel writes to a ring buffer (<code>dmesg</code>) and user-space daemons log through syslog/journald. "
    "Kernel messages reveal OS-level incidents: the <b>OOM killer</b> terminating processes, <b>segfaults</b> "
    "(invalid page access), <b>hung tasks</b> stuck in D state, disk <b>I/O errors</b> and <b>thermal throttling</b>. "
    "Logs are parsed, matched against OS event rules, and grouped into time windows for anomaly detection."
)


@st.cache_data(ttl=60, show_spinner="Reading logs...")
def load_logs(source: str, lines: int):
    return logs.load(source, lines=lines)


with st.sidebar:
    st.subheader("Source")
    avail = available_sources()
    st.caption("Available: " + ", ".join(k for k, v in avail.items() if v))
    source = st.selectbox("Log source", ["auto", "journald", "syslog", "dmesg", "sample", "upload"])
    lines = st.slider("Max lines", 500, 20000, 5000, 500)
    window = st.selectbox("Anomaly window", ["1min", "5min", "15min"], index=0)
    contamination = st.slider("Expected anomaly share", 0.01, 0.2, 0.05, 0.01)

if source == "upload":
    up = st.file_uploader("Log file (syslog, journalctl -o short-iso or dmesg output)", type=["log", "txt"])
    if up is None:
        st.stop()
    name, df = "upload", logs.parse_text(up.read().decode(errors="replace"))
else:
    name, df = load_logs(source, lines)

if df.empty:
    st.warning("No log lines could be read from this source (on WSL enable systemd for journald, or try 'dmesg' / "
               "'sample').")
    st.stop()

events = logs.detect_events(df)
anoms = logs.detect_anomalies(df, window, contamination)
sev = df["severity"].astype(str)

c = st.columns(5)
c[0].metric("Source", name)
c[1].metric("Lines", f"{len(df):,}")
c[2].metric("Errors (err+)", int(sev.isin(["emerg", "alert", "crit", "err"]).sum()))
c[3].metric("OS events", len(events))
c[4].metric("Anomalous windows", int(anoms["anomaly"].sum()) if not anoms.empty else 0)
if name == "sample":
    st.caption("Using the generated sample log (realistic syslog with injected incidents) because no real source "
               "was selected or readable.")

tab_time, tab_events, tab_sources, tab_raw, tab_drain, tab_corr, tab_loghub = st.tabs(
    ["Timeline & anomalies", "OS events", "Sources", "Log viewer", "Templates (Drain)", "Incident correlation",
     "Loghub benchmark"])

with tab_time:
    if anoms.empty:
        st.info("No timestamped lines to build a timeline.")
    else:
        a = anoms.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a["timestamp"], y=a["lines"], name="All lines", mode="lines",
                                 line=dict(color=SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=a["timestamp"], y=a["errors"], name="Errors", mode="lines",
                                 line=dict(color=SERIES[1], width=2)))
        flagged = a[a["anomaly"]]
        fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["lines"], name="Anomaly", mode="markers",
                                 marker=dict(size=12, color=STATUS["critical"], symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="%{x}<br>anomalous window<extra></extra>"))
        fig.update_layout(hovermode="x unified", yaxis_title=f"lines per {window}")
        show(style(fig, 360, "Log volume over time with anomalous windows"), "timeline")
        st.markdown("**Anomalous windows** (Isolation Forest on window features + robust z-score on errors)")
        cols = ["timestamp", "lines", "errors", "warnings", "critical", "os_events", "error_z", "iforest_score",
                "iforest_anomaly", "zscore_anomaly"]
        st.dataframe(flagged[cols].sort_values("iforest_score", ascending=False), hide_index=True,
                     width="stretch")
        pick = st.selectbox("Inspect window", flagged["timestamp"].astype(str).tolist()) if not flagged.empty else None
        if pick:
            start = pd.Timestamp(pick)
            end = start + pd.Timedelta(window)
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
            st.dataframe(sub[["timestamp", "source", "severity", "message"]], hide_index=True, width="stretch")

with tab_events:
    if events.empty:
        st.success("No OS-level incidents (OOM kills, segfaults, I/O errors, hung tasks...) found.")
    else:
        summary = logs.event_summary(events)
        fig = go.Figure(go.Bar(x=summary["count"], y=summary["category"], orientation="h",
                               marker=dict(color=SERIES[0], cornerradius=4), text=summary["count"],
                               textposition="outside", textfont=dict(color=INK["secondary"]),
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 60 + 32 * len(summary), "OS events by category"), "events")
        st.dataframe(summary, hide_index=True, width="stretch")
        cat = st.multiselect("Category", summary["category"].tolist())
        view = events[events["category"].isin(cat)] if cat else events
        st.dataframe(view[["timestamp", "category", "severity", "source", "target", "message"]],
                     hide_index=True, width="stretch")
        with st.expander("What these events mean"):
            for r in logs.RULES:
                st.markdown(f"- **{r.category}** - {r.meaning}")

with tab_sources:
    a, b = st.columns(2)
    with a:
        top = logs.top_sources(df, 12)
        fig = go.Figure(go.Bar(x=top["count"], y=top["source"], orientation="h",
                               marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 400, "Most active sources"), "src_all")
    with b:
        top_e = logs.top_sources(df, 12, errors_only=True)
        if top_e.empty:
            st.info("No error-level lines.")
        else:
            fig = go.Figure(go.Bar(x=top_e["count"], y=top_e["source"], orientation="h",
                                   marker=dict(color=SERIES[1], cornerradius=4),
                                   hovertemplate="%{y}: %{x}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            show(style(fig, 400, "Sources of error-level lines"), "src_err")
    st.dataframe(sev.value_counts().rename_axis("severity").reset_index(name="lines"), hide_index=True)

with tab_raw:
    c1, c2 = st.columns([1, 2])
    levels = c1.multiselect("Severity", logs.SEVERITY_ORDER, default=["emerg", "alert", "crit", "err", "warning"])
    q = c2.text_input("Search text (regex)")
    view = df[sev.isin(levels)] if levels else df
    if q:
        view = view[view["message"].str.contains(q, case=False, regex=True, na=False)]
    st.dataframe(view[["timestamp", "host", "source", "pid", "severity", "message"]].tail(2000),
                 hide_index=True, width="stretch", height=480)

with tab_drain:
    st.markdown("**Drain** recovers the print statements behind log lines in one pass: lines descend a fixed-depth "
                "parse tree (length → first tokens) and join the most similar group; differing tokens become `<*>`.")
    sim = st.slider("Similarity threshold", 0.3, 0.9, 0.5, 0.05)
    drain = logs.Drain(sim_threshold=sim)
    ids = drain.fit(df["message"].astype(str).tolist())
    tpl = drain.templates()
    c = st.columns(3)
    c[0].metric("Lines", f"{len(df):,}")
    c[1].metric("Templates", len(tpl))
    c[2].metric("Compression", f"{len(df) / max(len(tpl), 1):.0f}x")
    st.dataframe(tpl, hide_index=True, width="stretch", height=380)
    seq = pd.Series(drain.final_assignments())
    if len(seq) > 50:
        split = int(len(seq) * 0.6)
        model = logs.NGramSequenceModel(history=3, top_k=5).fit([seq[:split].tolist()])
        flags = model.anomalies(seq[split:].tolist())
        st.markdown(f"**Sequence model (DeepLog rule)** trained on the first 60% of lines: "
                    f"{sum(flags)} of {len(flags)} later lines were not among the top-5 predicted next templates.")
        odd = df.iloc[split:][pd.Series(flags, index=df.index[split:])]
        st.dataframe(odd[["timestamp", "source", "message"]].head(50), hide_index=True, width="stretch")

with tab_corr:
    st.markdown("Joins **what the system said** (OS events in the logs) with **what the metrics did** (robust-z jumps "
                "in recorded CPU, memory, swap and load) to build an incident timeline.")
    hist = get_recorder().store.system_history(24 * 60)
    if hist.empty:
        st.info("No recorded metrics yet.")
    else:
        res = logs.incident_timeline(hist, df)
        c = st.columns(3)
        c[0].metric("OS events", res["total_events"])
        c[1].metric("Metric anomalies", len(res["metric_anomalies"]))
        c[2].metric("Events explained by metrics", res["explained"])
        corr = res["correlated"]
        if corr.empty:
            st.success("No OS events to correlate.")
        else:
            st.dataframe(corr[["timestamp", "category", "related_metrics", "narrative"]], hide_index=True,
                         width="stretch")
        if not res["metric_anomalies"].empty:
            st.dataframe(res["metric_anomalies"].tail(50), hide_index=True, width="stretch")
        st.caption("Tip: run `python workloads/memory_leak.py` until memory pressure appears, then pick the journald "
                   "or dmesg source to see an OOM kill matched to the memory spike.")

with tab_loghub:
    import json
    from pathlib import Path

    st.markdown("Evaluation on the public **Loghub** 2k-line samples (BGL supercomputer, HDFS, Linux), which ship with "
                "ground-truth templates and, for BGL, alert labels.")
    saved = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "loghub.json"
    result = json.loads(saved.read_text()) if saved.exists() else None
    if st.button("Run evaluation now (downloads ~1 MB from GitHub)"):
        from kernelsense.logs import loghub
        with st.spinner("Parsing and evaluating..."):
            try:
                result = loghub.run_all()
            except OSError as exc:
                st.error(f"Download failed: {exc}")
    if result and "parsing" in result:
        st.markdown("**Drain grouping accuracy**")
        st.dataframe(pd.DataFrame([{"dataset": k, **v} for k, v in result["parsing"].items()]), hide_index=True,
                     width="stretch")
        b = result["bgl_anomaly"]
        st.markdown(f"**BGL anomaly detection** - {b['test_windows']} test windows of {b['window_lines']} lines "
                    f"({b['test_anomalous']} contain alerts)")
        st.dataframe(pd.DataFrame([{"method": "n-gram sequence (DeepLog rule)", **b["ngram_deeplog"]},
                                   {"method": "Isolation Forest", **b["isolation_forest"]}]), hide_index=True,
                     width="stretch")
