import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.paging import (
    ALL_POLICIES, BELADY_CLASSIC, PATTERNS, compare, detect_belady, fault_curve, generate, parse_refs, search_belady,
    thrashing_assessment, working_set,
)
from kernelsense.ui.common import INK, SERIES, STATUS, concept, get_recorder, page_setup, show, sidebar_status, style

page_setup("Page Replacement Simulation", "📄")
sidebar_status(get_recorder())

concept(
    "With demand paging, a reference to a page not in a physical frame causes a <b>page fault</b>. When all frames "
    "are full the OS must choose a <b>victim</b>. Optimal (Belady's MIN) evicts the page used farthest in the future - "
    "impossible in practice but the lower bound. LRU uses the past to approximate it; Clock approximates LRU with one "
    "hardware reference bit per frame. FIFO can suffer <b>Belady's anomaly</b>: more frames, more faults."
)

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
mode = c1.radio("Reference string", ["Generate", "Custom", "Algorithmic trace"], horizontal=True)
frames = c2.number_input("Frames", 1, 64, 3 if mode != "Algorithmic trace" else 8)
if mode == "Algorithmic trace":
    from kernelsense.paging import traces

    tname = c3.selectbox("Program", list(traces.ALGORITHMIC), format_func=lambda k: traces.ALGORITHMIC[k][0])
    limit = c4.number_input("Max references", 100, 20000, 5000, 500)
    refs = traces.algorithmic(tname, int(limit))
    st.caption(f"{traces.ALGORITHMIC[tname][0]}: pages computed from array element addresses (8-byte elements, "
               f"4 KiB pages) · {len(refs)} references over {len(set(refs))} distinct pages.")
elif mode == "Generate":
    pattern = c3.selectbox("Pattern", list(PATTERNS), format_func=lambda p: p, index=0)
    length = c4.number_input("Length", 5, 200, 20)
    refs = generate(pattern, int(length), 10, seed=7)
    st.caption(PATTERNS[pattern])
else:
    text = c3.text_input("Pages (space/comma separated)", "7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1")
    try:
        refs = parse_refs(text)
    except ValueError as e:
        st.error(str(e))
        st.stop()
st.code(" ".join(map(str, refs[:400])) + (" ..." if len(refs) > 400 else ""), language=None)

tab_step, tab_cmp, tab_curve, tab_belady, tab_ws, tab_mattson = st.tabs(
    ["Step-by-step", "Comparison", "Faults vs frames", "Belady's anomaly", "Working set & thrashing",
     "Miss-ratio curve (Mattson)"])

results = {k: cls().simulate(refs, int(frames)) for k, cls in ALL_POLICIES.items()}

with tab_step:
    algo = st.radio("Algorithm", list(ALL_POLICIES), horizontal=True)
    r = results[algo]
    c = st.columns(4)
    c[0].metric("Page faults", r.faults)
    c[1].metric("Hits", r.hits)
    c[2].metric("Fault rate", f"{r.fault_rate:.1%}")
    c[3].metric("Hit ratio", f"{1 - r.fault_rate:.1%}")
    if len(refs) > 80:
        st.caption("Showing the first 80 references; metrics above cover the full trace.")
        r_show = ALL_POLICIES[algo]().simulate(refs[:80], int(frames))
    else:
        r_show = r
    table = r_show.table()

    def color_result(v):
        if v == "F":
            return f"background-color: {STATUS['critical']}; color: white; font-weight: 600"
        if v == "H":
            return f"background-color: {STATUS['good']}; color: white; font-weight: 600"
        return ""

    st.dataframe(table.style.map(color_result), width="stretch")
    steps = pd.DataFrame([{"#": s.index + 1, "page": s.page, "result": "Hit" if s.hit else "Fault",
                           "evicted": "" if s.evicted is None else s.evicted,
                           "frames": " | ".join("-" if f is None else str(f) for f in s.frames),
                           **({"ref bits": s.extra.get("bits"), "hand": s.extra.get("hand")} if algo == "Clock" else {}),
                           **({"frequency": s.extra.get("freq")} if algo == "LFU" else {})}
                          for s in r_show.steps])
    with st.expander("Victim selection log"):
        st.dataframe(steps, hide_index=True, width="stretch")

with tab_cmp:
    summ = pd.DataFrame([x.summary() for x in results.values()])
    best = summ["faults"].min()
    fig = go.Figure(go.Bar(x=summ["algorithm"], y=summ["faults"],
                           marker=dict(color=[SERIES[2] if v == best else SERIES[0] for v in summ["faults"]],
                                       cornerradius=4),
                           text=summ["faults"], textposition="outside", textfont=dict(color=INK["secondary"]),
                           hovertemplate="%{x}: %{y} faults<extra></extra>"))
    fig.update_yaxes(title="page faults", rangemode="tozero")
    show(style(fig, 340, f"Page faults with {frames} frames (fewest highlighted)"), "cmp")
    st.dataframe(summ, hide_index=True, width="stretch")

with tab_curve:
    maxf = st.slider("Max frames", 2, 40, 8)
    curve = fault_curve(refs, maxf)
    fig = go.Figure()
    for i, name in enumerate(ALL_POLICIES):
        d = curve[curve["algorithm"] == name]
        fig.add_trace(go.Scatter(x=d["frames"], y=d["faults"], name=name, mode="lines+markers",
                                 line=dict(color=SERIES[i], width=2), marker=dict(size=8)))
    fig.update_layout(hovermode="x unified", xaxis=dict(title="frames", dtick=1), yaxis_title="page faults")
    show(style(fig, 380, "Page faults as frames increase"), "curve")
    st.caption("A healthy policy's curve never rises. LRU and Optimal are stack algorithms and can never show "
               "Belady's anomaly; FIFO can.")

with tab_belady:
    a, b = st.columns(2)
    with a:
        st.markdown("**Classic example** `1 2 3 4 1 2 5 1 2 3 4 5`")
        rows = [{"frames": f, "FIFO faults": ALL_POLICIES["FIFO"]().simulate(BELADY_CLASSIC, f).faults,
                 "LRU faults": ALL_POLICIES["LRU"]().simulate(BELADY_CLASSIC, f).faults} for f in range(1, 7)]
        bdf = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bdf["frames"], y=bdf["FIFO faults"], name="FIFO", mode="lines+markers",
                                 line=dict(color=SERIES[0], width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=bdf["frames"], y=bdf["LRU faults"], name="LRU", mode="lines+markers",
                                 line=dict(color=SERIES[1], width=2), marker=dict(size=8)))
        fig.add_annotation(x=4, y=10, text="4 frames → 10 faults<br>(3 frames → 9)", showarrow=True,
                           arrowcolor=INK["muted"], font=dict(color=INK["secondary"]))
        fig.update_layout(xaxis=dict(title="frames", dtick=1), yaxis_title="faults", hovermode="x unified")
        show(style(fig, 330, "Belady's anomaly under FIFO"), "belady")
    with b:
        st.markdown("**Your reference string**")
        cases = detect_belady(refs, max_frames=10)
        if cases:
            for cs in cases:
                st.error(f"Anomaly: {cs.frames_low} frames → {cs.faults_low} faults, "
                         f"{cs.frames_high} frames → {cs.faults_high} faults")
        else:
            st.success("No Belady's anomaly for FIFO on this string (frames 1-10).")
        if st.button("Search for a new anomalous string"):
            seed = st.session_state.get("bseed", 0) + 1
            st.session_state.bseed = seed
            found = search_belady(seed=seed)
            if found:
                st.code(" ".join(map(str, found.refs)), language=None)
                st.write(f"FIFO: {found.frames_low} frames → {found.faults_low} faults, "
                         f"{found.frames_high} frames → {found.faults_high} faults")
            else:
                st.info("None found in this batch, try again.")

with tab_ws:
    window = st.slider("Working-set window Δ (references)", 2, 30, 10)
    ws = working_set(refs, window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(ws) + 1)), y=ws, name="Working-set size", mode="lines",
                             line=dict(color=SERIES[0], width=2, shape="hv")))
    fig.add_hline(y=frames, line=dict(color=STATUS["critical"], dash="dot"), annotation_text=f"{frames} frames",
                  annotation_font_color=INK["secondary"])
    fig.update_layout(xaxis_title="time (reference #)", yaxis_title="distinct pages in window")
    show(style(fig, 320, "Working set W(t, Δ) vs allocated frames"), "ws")
    t = thrashing_assessment(refs, int(frames), window)
    c = st.columns(3)
    c[0].metric("Average working set", t["avg_working_set"])
    c[1].metric("Peak working set", t["peak_working_set"])
    c[2].metric("Thrashing risk", t["thrashing_risk"])
    st.caption(t["explanation"])

with tab_mattson:
    from kernelsense.paging import mattson_lru_curve, reuse_distance_histogram

    st.markdown("**Mattson's stack algorithm**: LRU is a stack algorithm, so a reference with stack (reuse) distance "
                "*d* hits exactly when memory has ≥ *d* frames. One pass over the trace gives the LRU fault count for "
                "**every** memory size - the miss-ratio curve OS designers use to size caches.")
    maxc = st.slider("Frame sizes", 2, 64, 32, key="mattson_max")
    curve = mattson_lru_curve(refs, maxc)
    fig = go.Figure(go.Scatter(x=curve["frames"], y=curve["miss_ratio"], mode="lines+markers",
                               line=dict(color=SERIES[0], width=2), marker=dict(size=7),
                               hovertemplate="%{x} frames: %{y:.1%} misses<extra></extra>"))
    fig.update_layout(xaxis_title="frames", yaxis=dict(title="LRU miss ratio", tickformat=".0%"))
    show(style(fig, 340, "LRU miss-ratio curve from one pass"), "mrc")
    hist = reuse_distance_histogram(refs)
    st.dataframe(hist.head(20), hide_index=True, width="stretch")
