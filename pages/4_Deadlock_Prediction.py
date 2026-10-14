import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.deadlock import (
    BankersState, ResourceAllocationGraph, bankers_risk, dining_philosophers, graph_risk, live_report,
    multi_instance_no_deadlock, simple_deadlock, textbook_example, unsafe_example,
)
from kernelsense.ui.common import INK, SERIES, STATUS, concept, get_recorder, page_setup, show, sidebar_status, style

page_setup("Deadlock Prediction", "🔒")
sidebar_status(get_recorder())

concept(
    "A deadlock needs all four <b>Coffman conditions</b>: mutual exclusion, hold-and-wait, no preemption and "
    "circular wait. <b>Avoidance</b> (Banker's algorithm) only grants a request if the system stays in a safe state; "
    "<b>detection</b> looks for cycles in the wait-for graph. The risk score predicts trouble <i>before</i> a cycle "
    "closes by measuring how close the system is to these conditions."
)


def risk_panel(report):
    color = {"Low": "good", "Medium": "warning", "High": "serious", "Critical": "critical"}[report.level]
    icon = {"good": "🟢", "warning": "🟠", "serious": "🟠", "critical": "🔴"}[color]
    st.metric(f"Deadlock risk · {icon} {report.level}", f"{report.score:.0f}/100")
    fig = go.Figure(go.Bar(x=[report.score], y=["risk"], orientation="h",
                           marker=dict(color=STATUS[color], cornerradius=4), hoverinfo="skip"))
    fig.update_xaxes(range=[0, 100], showgrid=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    show(style(fig, 70), f"risk_{id(report)}")
    for n in report.notes:
        st.markdown(f"- {n}")
    st.caption("Factors: " + ", ".join(f"{k} = {v}" for k, v in report.factors.items()))


def matrix_df(mat, rows, cols):
    return pd.DataFrame(mat, index=rows, columns=cols)


tab_bank, tab_rag, tab_lockdep, tab_live = st.tabs(["Banker's algorithm (avoidance)", "Resource allocation graph (detection)",
                                                    "Lock-order validation (lockdep)", "Live system"])

# --------------------------------------------------------------- Banker's
with tab_bank:
    presets = {"Textbook (Silberschatz)": textbook_example, "Unsafe state": unsafe_example}
    c1, c2, c3 = st.columns([2, 1, 1])
    preset = c1.selectbox("Preset", list(presets) + ["Custom size"])
    if preset == "Custom size":
        n = c2.number_input("Processes", 2, 10, 4)
        m = c3.number_input("Resource types", 1, 6, 3)
        procs = [f"P{i}" for i in range(int(n))]
        res = [chr(65 + j) for j in range(int(m))]
        base = BankersState(procs, res, [3] * int(m), [[2] * int(m)] * int(n), [[0] * int(m)] * int(n))
    else:
        base = presets[preset]()

    a, b = st.columns(2)
    with a:
        st.markdown("**Allocation** (currently held)")
        alloc = st.data_editor(matrix_df(base.allocation, base.processes, base.resources), key=f"al_{preset}",
                               width="stretch")
        st.markdown("**Available**")
        avail = st.data_editor(pd.DataFrame([base.available], columns=base.resources, index=["Available"]),
                               key=f"av_{preset}", width="stretch")
    with b:
        st.markdown("**Max** (declared maximum demand)")
        maxm = st.data_editor(matrix_df(base.maximum, base.processes, base.resources), key=f"mx_{preset}",
                              width="stretch")
    try:
        state = BankersState(base.processes, base.resources, [int(x) for x in avail.iloc[0].tolist()],
                             maxm.astype(int).values.tolist(), alloc.astype(int).values.tolist())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with b:
        st.markdown("**Need = Max − Allocation**")
        st.dataframe(matrix_df(state.need, state.processes, state.resources), width="stretch")

    safety = state.safety()
    left, right = st.columns([3, 2])
    with left:
        if safety.safe:
            st.success(f"✅ SAFE · safe sequence: **⟨{', '.join(safety.sequence)}⟩**")
        else:
            st.error(f"⛔ UNSAFE · cannot guarantee completion of: **{', '.join(safety.blocked)}**")
        st.markdown("**Safety algorithm trace**")
        st.dataframe(pd.DataFrame([{"step": i + 1, "process": s.process, "need": s.need, "work before": s.work_before,
                                    "work after (+ allocation released)": s.work_after}
                                   for i, s in enumerate(safety.steps)]), hide_index=True, width="stretch")
    with right:
        risk_panel(bankers_risk(state))

    st.markdown("#### Resource request")
    c1, c2, c3 = st.columns([1, 2, 1])
    who = c1.selectbox("Process", state.processes)
    vec = c2.text_input("Request vector", " ".join("0" for _ in state.resources), help="One number per resource type")
    if c3.button("Request", width="stretch"):
        try:
            r = state.request(who, [int(x) for x in vec.split()])
            (st.success if r.granted else st.warning)(r.reason)
            if r.new_state is not None and r.granted:
                st.caption("State after granting:")
                st.dataframe(matrix_df(r.new_state.allocation, state.processes, state.resources),
                             width="stretch")
        except ValueError:
            st.error("Request must be integers, one per resource type.")

# --------------------------------------------------------------------- RAG
with tab_rag:
    presets = {
        "Two processes, two resources (deadlock)": simple_deadlock,
        "Dining philosophers (deadlock)": dining_philosophers,
        "Cycle without deadlock (multi-instance)": multi_instance_no_deadlock,
        "Custom": None,
    }
    choice = st.selectbox("Scenario", list(presets))
    if presets[choice]:
        g = presets[choice]()
        res_df = pd.DataFrame({"resource": list(g.instances), "instances": list(g.instances.values())})
        held_df = pd.DataFrame([{"process": p, "resource": r, "count": c} for (p, r), c in g.held.items()])
        req_df = pd.DataFrame([{"process": p, "resource": r, "count": c} for (p, r), c in g.requested.items()])
    else:
        res_df = pd.DataFrame({"resource": ["R1", "R2", "R3"], "instances": [1, 1, 1]})
        held_df = pd.DataFrame({"process": ["P1", "P2"], "resource": ["R1", "R2"], "count": [1, 1]})
        req_df = pd.DataFrame({"process": ["P1"], "resource": ["R2"], "count": [1]})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Resources**")
        res_df = st.data_editor(res_df, num_rows="dynamic", key=f"r_{choice}", width="stretch")
    with c2:
        st.markdown("**Assignment edges** (R → P, held)")
        held_df = st.data_editor(held_df, num_rows="dynamic", key=f"h_{choice}", width="stretch")
    with c3:
        st.markdown("**Request edges** (P → R, waiting)")
        req_df = st.data_editor(req_df, num_rows="dynamic", key=f"q_{choice}", width="stretch")

    try:
        g = ResourceAllocationGraph()
        for row in res_df.dropna().itertuples(index=False):
            g.add_resource(str(row.resource), int(row.instances))
        for row in held_df.dropna().itertuples(index=False):
            g.assign(str(row.resource), str(row.process), int(row.count))
        for row in req_df.dropna().itertuples(index=False):
            g.request(str(row.process), str(row.resource), int(row.count))
    except (ValueError, TypeError) as e:
        st.error(f"Invalid graph: {e}")
        st.stop()

    det = g.detect()
    left, right = st.columns([3, 2])
    with left:
        G = g.graph()
        pos = nx.circular_layout(G) if len(G) <= 12 else nx.spring_layout(G, seed=1)
        dead = set(det.deadlocked_processes)
        fig = go.Figure()
        for u, v, d in G.edges(data=True):
            color = SERIES[0] if d["kind"] == "assignment" else SERIES[1]
            fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2,
                               arrowcolor=color, standoff=18, startstandoff=18)
        for kind, symbol in (("process", "circle"), ("resource", "square")):
            nodes = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
            if not nodes:
                continue
            colors = [STATUS["critical"] if n in dead else ("#e1e0d9" if kind == "resource" else "#cde2fb")
                      for n in nodes]
            labels = [f"{n}<br>×{G.nodes[n]['instances']}" if kind == "resource" and G.nodes[n]["instances"] > 1
                      else n for n in nodes]
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                                     marker=dict(symbol=symbol, size=42, color=colors, line=dict(color=INK["axis"], width=2)),
                                     text=labels, textfont=dict(color=INK["primary"], size=11), name=kind.title() + "s",
                                     hovertemplate="%{text}<extra></extra>"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        show(style(fig, 460, "Resource allocation graph · blue = held (R→P), orange = requested (P→R)"), "rag")
    with right:
        if det.deadlocked:
            st.error(f"⛔ DEADLOCK · {', '.join(det.deadlocked_processes)}")
        else:
            st.success("✅ No deadlock")
        st.caption(f"Method: {det.method}")
        if det.cycles:
            st.markdown("**Cycles found:** " + "; ".join(" → ".join(c + [c[0]]) for c in det.cycles[:5]))
        if det.finish_order:
            st.markdown(f"**Reduction order:** {' → '.join(det.finish_order)}")
        risk_panel(graph_risk(g))
        wfg = g.wait_for_graph()
        if wfg.number_of_edges():
            st.markdown("**Wait-for graph edges**")
            st.dataframe(pd.DataFrame([{"waiting": u, "waits for": v, "resource": d["resource"]}
                                       for u, v, d in wfg.edges(data=True)]), hide_index=True, width="stretch")
    st.caption("Single-instance resources: a cycle ⇔ deadlock. Multiple instances: a cycle is necessary but not "
               "sufficient, so the graph-reduction detection algorithm decides.")

# -------------------------------------------------------------------- live
with tab_live:
    rep = live_report()
    c = st.columns(4)
    c[0].metric("File locks held", rep["held"])
    c[1].metric("Waiting on locks", rep["waiting"])
    c[2].metric("Lock wait cycles", len(rep["cycles"]))
    c[3].metric("Tasks in D state", len(rep["d_state"]))
    if rep["deadlock"]:
        st.error("⛔ Circular wait on file locks: " + "; ".join(" → ".join(map(str, c)) for c in rep["cycles"]))
    else:
        st.success("✅ No circular wait among file-lock holders.")
    if rep["locks"]:
        st.dataframe(pd.DataFrame([l.__dict__ for l in rep["locks"]]), hide_index=True, width="stretch")
    else:
        st.info("No POSIX/flock locks currently held. Run `python workloads/lock_deadlock.py` to create a real "
                "file-lock deadlock between two processes and watch it appear here.")
    if rep["d_state"]:
        st.warning("Processes in uninterruptible sleep: " + ", ".join(f"{d['name']} ({d['pid']})" for d in rep["d_state"]))
    st.caption("Source: /proc/locks (lines with '->' are waiters blocked on the lock above them) and process states.")

with tab_lockdep:
    from kernelsense.deadlock.lockdep import demo_dining_philosophers, demo_inversion
    from kernelsense.deadlock.seeded import benchmark

    st.markdown("Inspired by the Linux kernel's **lockdep**: every time a thread takes lock B while holding A, the edge "
                "A → B is added to a global lock-order graph. If a new edge would close a cycle, a deadlock is "
                "*possible* - reported even though this run did not hang.")
    demo = st.radio("Demo", ["Inverted lock order", "Dining philosophers", "Dining philosophers with resource ordering"],
                    horizontal=True)
    mon = {"Inverted lock order": demo_inversion,
           "Dining philosophers": demo_dining_philosophers,
           "Dining philosophers with resource ordering": lambda: demo_dining_philosophers(ordered=True)}[demo]()
    rep = mon.report()
    a, b = st.columns([3, 2])
    with a:
        G = mon.graph
        pos = nx.circular_layout(G)
        cyc_nodes = {n for v in mon.violations for n in v.cycle}
        fig = go.Figure()
        for u, v in G.edges():
            fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y", axref="x",
                               ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2,
                               arrowcolor=SERIES[1] if (u in cyc_nodes and v in cyc_nodes) else SERIES[0],
                               standoff=14, startstandoff=14)
        nodes = list(G.nodes())
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                                 text=nodes, textposition="bottom center", marker=dict(size=30, symbol="square",
                                                         color=[STATUS["critical"] if n in cyc_nodes else "#cde2fb"
                                                                for n in nodes], line=dict(color=INK["axis"], width=2)),
                                 textfont=dict(size=10, color=INK["primary"]), hovertemplate="%{text}<extra></extra>"))
        fig.update_xaxes(visible=False, range=[-1.4, 1.4])
        fig.update_yaxes(visible=False, scaleanchor="x")
        show(style(fig, 400, "Lock-order graph (edge A→B: B acquired while holding A)"), "lockgraph")
    with b:
        st.metric("Lock acquisitions", rep["acquisitions"])
        st.metric("Potential deadlocks", rep["violations"])
        for d in rep["details"]:
            st.error(d)
        if not rep["details"]:
            st.success("Consistent global lock order - no deadlock possible.")
    st.markdown("#### Seeded-bug benchmark")
    n = st.slider("Programs (half contain a lock-order inversion)", 20, 300, 100, 20)
    if st.button("Run benchmark"):
        with st.spinner("Running generated programs..."):
            r = benchmark(n)
        c = st.columns(3)
        c[0].metric("lockdep detection rate", f"{r['lockdep_detection_rate']:.0%}")
        c[1].metric("False positives", f"{r['lockdep_false_positive_rate']:.0%}")
        c[2].metric("Bugs that actually deadlocked in testing", f"{r['observed_deadlock_rate_buggy']:.0%}")
        st.caption("Short critical sections almost never hit the bad interleaving in testing, so ordinary testing "
                   "misses these bugs; lock-order validation finds all of them without needing the deadlock to happen.")
    st.code("""from kernelsense.deadlock import LockOrderMonitor, patch_threading
mon = LockOrderMonitor()
with patch_threading(mon):      # threading.Lock()/RLock() become tracked locks
    run_your_program()
print(mon.report())""", language="python")
