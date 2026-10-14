"""Shared Streamlit helpers: recorder singleton, chart styling and reusable charts."""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from ..recorder import Recorder
from ..storage import MetricStore

# Validated categorical order (fixed, never cycled) and reserved status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SURFACE = "#fcfcfb"
NEUTRAL = "#b7b5ac"

SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵", "good": "🟢"}

pio.templates["kernelsense"] = go.layout.Template(
    layout=go.Layout(
        colorway=SERIES,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK["secondary"], size=13),
        title=dict(font=dict(color=INK["primary"], size=15), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False, tickfont=dict(color=INK["muted"]),
                   automargin=True),
        yaxis=dict(gridcolor=INK["grid"], linecolor=INK["axis"], zeroline=False, tickfont=dict(color=INK["muted"]),
                   automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        margin=dict(l=56, r=16, t=64, b=44),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK["primary"])),
        bargap=0.25,
    )
)
pio.templates.default = "kernelsense"


def page_setup(title: str, icon: str = "🧠") -> None:
    st.set_page_config(page_title=f"{title} · KernelSense", page_icon=icon, layout="wide")
    st.markdown(
        """<style>
        .block-container {padding-top: 1.6rem;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .ks-concept {border-left: 3px solid #2a78d6; padding: .4rem .8rem; background: #f0efec;
                     border-radius: 4px; font-size: .92rem; color: #52514e;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title(title)


def concept(text: str) -> None:
    """OS concept callout shown at the top of each module (useful during demos/viva)."""
    st.markdown(f"<div class='ks-concept'>{text}</div>", unsafe_allow_html=True)


@st.cache_resource
def get_recorder() -> Recorder:
    """One recorder thread per Streamlit server, shared by every page and session."""
    rec = Recorder(MetricStore())
    rec.start()
    # wait briefly so the first page render has data
    for _ in range(30):
        if rec.latest_system is not None:
            break
        time.sleep(0.1)
    return rec


def sidebar_status(rec: Recorder) -> None:
    with st.sidebar:
        st.caption("Recorder")
        st.write(f"{'🟢 running' if rec.running else '🔴 stopped'} · every {rec.interval:g}s")
        st.write(f"Samples: {rec.samples} · overhead {rec.overhead_ms:.0f} ms")
        st.caption(f"DB: {rec.store.db_path.name}")


def style(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, template="kernelsense")
    if title:
        fig.update_layout(title=title)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


def color_map(keys) -> dict:
    """Stable entity -> colour mapping. More than 8 entities fall back to one neutral hue + labels."""
    keys = list(dict.fromkeys(keys))
    if len(keys) <= len(SERIES):
        return {k: SERIES[i] for i, k in enumerate(keys)}
    return {k: SERIES[0] for k in keys}


def line_chart(df: pd.DataFrame, x: str, ys: dict[str, str], title: str, y_title: str = "",
               height: int = 300, y_range=None) -> go.Figure:
    fig = go.Figure()
    for i, (col, label) in enumerate(ys.items()):
        fig.add_trace(go.Scatter(x=df[x], y=df[col], name=label, mode="lines",
                                 line=dict(width=2, color=SERIES[i])))
    fig.update_layout(hovermode="x unified", yaxis_title=y_title, showlegend=len(ys) > 1)
    if y_range:
        fig.update_yaxes(range=y_range)
    return style(fig, height, title)


def gantt(segments: pd.DataFrame, title: str = "Gantt chart", height: int | None = None) -> go.Figure:
    """Horizontal Gantt chart; IDLE gaps drawn in neutral grey, every bar labelled with its PID."""
    pids = [p for p in segments["pid"].unique() if p != "IDLE"]
    cmap = color_map(pids)
    cmap["IDLE"] = "#e1e0d9"
    fig = go.Figure()
    shown = set()
    for row in segments.itertuples(index=False):
        fig.add_trace(go.Bar(
            x=[row.end - row.start], base=[row.start], y=["CPU"], orientation="h",
            marker=dict(color=cmap[row.pid], line=dict(color=SURFACE, width=2)),
            name=row.pid, legendgroup=row.pid, showlegend=row.pid not in shown and len(pids) <= 8,
            text=row.pid if (row.end - row.start) >= 1 else "", textposition="inside",
            insidetextanchor="middle", textfont=dict(color="white" if row.pid != "IDLE" else INK["muted"]),
            hovertemplate=f"<b>{row.pid}</b><br>{row.start} → {row.end} ({row.end-row.start} units)<extra></extra>",
        ))
        shown.add(row.pid)
    end = int(segments["end"].max()) if not segments.empty else 0
    step = max(1, end // 25)
    fig.update_layout(barmode="overlay", xaxis=dict(title="time", tickmode="linear", dtick=step, range=[0, max(end, 1)]),
                      yaxis=dict(showticklabels=False), bargap=0.1)
    return style(fig, height or 170, title)


def df_download(df: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv")
