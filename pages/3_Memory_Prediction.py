import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.config import settings
from kernelsense.memory import default_models, detect_leaks, evaluate, regularize, time_to_threshold
from kernelsense.ui.common import INK, SERIES, STATUS, concept, get_recorder, page_setup, show, sidebar_status, style

page_setup("Memory Usage Prediction", "📈")
rec = get_recorder()
sidebar_status(rec)

concept(
    "Virtual memory lets processes use more address space than physical RAM. When free RAM runs out the kernel "
    "reclaims page cache, swaps, and finally invokes the <b>OOM killer</b>. Forecasting memory usage gives an early "
    "warning before that happens; a process whose resident set (RSS) grows steadily and never plateaus is a "
    "<b>memory leak</b> suspect."
)


def forecast_chart(series: pd.Series, fc, title: str, y_title: str, threshold: float | None, interval_s: float):
    future = pd.date_range(series.index[-1] + pd.Timedelta(seconds=interval_s), periods=len(fc.mean),
                           freq=f"{max(int(interval_s), 1)}s")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Observed", mode="lines",
                             line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(fc.upper) + list(fc.lower[::-1]),
                             fill="toself", fillcolor="rgba(235,104,52,0.15)", line=dict(width=0),
                             name="95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future, y=fc.mean, name=f"Forecast ({fc.model})", mode="lines",
                             line=dict(color=SERIES[1], width=2, dash="dash")))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=STATUS["critical"], width=1, dash="dot"),
                      annotation_text=f"alert {threshold:g}", annotation_font_color=INK["secondary"])
    fig.update_layout(hovermode="x unified", yaxis_title=y_title)
    return style(fig, 380, title)


@st.cache_data(show_spinner=False, ttl=30)
def cached_eval(values: tuple) -> pd.DataFrame:
    return evaluate(np.array(values))


def demo_series(n: int = 240, interval: float = 2.0) -> pd.Series:
    rng = np.random.default_rng(3)
    base = 45 + 0.06 * np.arange(n) + 2 * np.sin(np.arange(n) / 15) + rng.normal(0, 0.6, n)
    idx = pd.date_range(pd.Timestamp.now() - pd.Timedelta(seconds=n * interval), periods=n, freq=f"{int(interval)}s")
    return pd.Series(base, index=idx)


tab_sys, tab_leak, tab_proc, tab_oom = st.tabs(["System memory forecast", "Leak detection", "Process forecast",
                                               "OOM victim prediction"])

with tab_sys:
    s, _ = rec.snapshot()
    if s:
        c = st.columns(5)
        c[0].metric(f"Used ({s.mem_percent:.1f}%)", f"{s.mem_used_mb/1024:.2f} GB")
        c[1].metric("Available", f"{s.mem_available_mb/1024:.2f} GB")
        c[2].metric("Page cache", f"{s.mem_cached_mb/1024:.2f} GB")
        c[3].metric(f"Swap used ({s.swap_percent:.1f}%)", f"{s.swap_used_mb/1024:.2f} GB")
        psi = (s.pressure or {}).get("memory") or {}
        c[4].metric("Memory pressure (PSI)", f"{psi.get('some', {}).get('avg10', 0):.2f}%" if psi else "n/a")

    c1, c2, c3, c4 = st.columns(4)
    window = c1.select_slider("Training window", [10, 30, 60, 180, 720], 60, format_func=lambda m: f"{m} min")
    horizon_min = c2.slider("Forecast horizon (min)", 1, 120, 15)
    threshold = c3.number_input("Alert threshold (%)", 10.0, 100.0, float(settings.memory_alert_pct))
    use_demo = c4.toggle("Use demo series", value=False, help="Synthetic growing-memory series for demonstrations.")

    hist = rec.store.system_history(window)
    interval = rec.interval
    if use_demo or len(hist) < 30:
        if not use_demo:
            st.info(f"Only {len(hist)} samples recorded so far (need 30). Showing a demo series until enough "
                    "real history is collected - keep the dashboard or `kernelsense record` running.")
        series = demo_series(interval=2.0)
        interval = 2.0
    else:
        series = regularize(hist, "mem_percent", interval)

    steps = max(1, int(horizon_min * 60 / interval))
    table = cached_eval(tuple(np.round(series.values, 4)))
    best_name = table.dropna(subset=["MAE"]).iloc[0]["model"]
    model = next(m for m in default_models() if m.name == best_name).fit(series.values)
    fc = model.predict(steps)
    eta = time_to_threshold(fc, threshold, interval)

    a, b = st.columns([3, 1])
    with a:
        show(forecast_chart(series, fc, "Memory utilisation forecast", "%", threshold, interval), "sysfc")
    with b:
        st.metric("Best model", best_name)
        st.metric(f"Predicted in {horizon_min} min", f"{fc.mean[-1]:.1f}%",
                  f"{fc.mean[-1] - series.values[-1]:+.1f} pts", delta_color="inverse")
        if eta is not None:
            st.error(f"⚠️ Threshold {threshold:g}% expected in **{eta/60:.1f} min**")
        else:
            st.success(f"✅ Not expected to cross {threshold:g}% within {horizon_min} min")
    st.markdown("**Model evaluation on held-out data** (last 20% of the window; a model must beat the naive baseline)")
    st.dataframe(table.round(3), hide_index=True, width="stretch")

with tab_leak:
    c1, c2, c3 = st.columns(3)
    minutes = c1.select_slider("Window", [5, 15, 30, 60, 180], 30, format_func=lambda m: f"{m} min")
    min_slope = c2.number_input("Min growth (MB/min)", 0.1, 100.0, 1.0)
    min_r2 = c3.slider("Min trend fit (R²)", 0.0, 1.0, 0.8)
    ph = rec.store.process_history(minutes=minutes)
    findings = detect_leaks(ph, min_slope_mb_per_min=min_slope, min_r2=min_r2)
    if not findings:
        st.info("Not enough per-process history yet. Run `python workloads/memory_leak.py` to see a leak detected.")
    else:
        fdf = pd.DataFrame([f.__dict__ for f in findings])
        sus = fdf[fdf["suspected"]]
        if sus.empty:
            st.success("No process shows a steady, monotonic RSS growth pattern.")
        else:
            st.error(f"🔴 {len(sus)} suspected leak(s): " + ", ".join(f"{r.name} ({r.pid})" for r in sus.itertuples()))
        st.dataframe(fdf, hide_index=True, width="stretch",
                     column_config={"suspected": st.column_config.CheckboxColumn("leak?")})
        top = fdf.head(5)["pid"].tolist()
        sub = ph[ph["pid"].isin(top)]
        fig = go.Figure()
        for i, pid in enumerate(top):
            d = sub[sub["pid"] == pid]
            name = d["name"].iloc[-1] if not d.empty else str(pid)
            fig.add_trace(go.Scatter(x=d["time"], y=d["rss_mb"], name=f"{name} ({pid})", mode="lines",
                                     line=dict(color=SERIES[i], width=2)))
        fig.update_layout(hovermode="x unified", yaxis_title="RSS MB")
        show(style(fig, 340, "Resident memory of the top candidates"), "leakchart")
        st.caption("Criteria: positive slope above the minimum, R² above the minimum (steady growth), and at least 70% "
                   "of consecutive samples non-decreasing.")

with tab_proc:
    ph = rec.store.process_history(minutes=60)
    if ph.empty:
        st.info("No process history recorded yet.")
    else:
        counts = ph.groupby("pid").size()
        eligible = counts[counts >= 15].index.tolist()
        if not eligible:
            st.info("Processes need at least 15 samples for a forecast.")
        else:
            names = ph.groupby("pid")["name"].last()
            pid = st.selectbox("Process", eligible, format_func=lambda p: f"{p} - {names[p]}")
            d = ph[ph["pid"] == pid]
            series = regularize(d, "rss_mb", rec.interval)
            h = st.slider("Horizon (min)", 1, 60, 10, key="ph")
            table = cached_eval(tuple(np.round(series.values, 4)))
            best = table.dropna(subset=["MAE"]).iloc[0]["model"]
            m = next(x for x in default_models() if x.name == best).fit(series.values)
            fc = m.predict(max(1, int(h * 60 / rec.interval)))
            show(forecast_chart(series, fc, f"RSS forecast - {names[pid]} ({pid})", "MB", None, rec.interval), "pfc")
            st.dataframe(table.round(3), hide_index=True, width="stretch")

with tab_oom:
    from kernelsense.memory import oom

    st.markdown("When RAM and swap run out, the kernel kills the process with the highest **badness**. "
                "`/proc/[pid]/oom_score` = (1000 + badness × 1000 / totalpages) × 2/3, where badness = RSS + swap + "
                "page tables (+ oom_score_adj). The estimate column recomputes it from `/proc/[pid]/status`.")
    cands = oom.frame(15)
    if cands.empty:
        st.info("OOM scores are not available on this system.")
    else:
        match = (abs(cands["oom_score"] - cands["estimated_score"]) <= 1).mean()
        hist = rec.store.system_history(60)
        eta = None
        if len(hist) >= 30:
            ser = regularize(hist, "mem_percent", rec.interval)
            m = default_models(include_ml=False)[3].fit(ser.values)
            eta = time_to_threshold(m.predict(int(3600 / rec.interval)), 99.0, rec.interval)
        pred = oom.prediction(eta)
        (st.error if eta else st.info)(pred["message"])
        st.metric("Estimate matches kernel oom_score (±1)", f"{match:.0%}")
        st.dataframe(cands, hide_index=True, width="stretch")
        fig = go.Figure(go.Bar(x=cands["oom_score"], y=[f"{n} ({p})" for n, p in zip(cands["name"], cands["pid"])],
                               orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 60 + 26 * len(cands), "OOM-killer ranking (highest is killed first)"), "oomrank")
