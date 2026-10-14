import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kernelsense.allocator import (
    Actuator, PolicyConfig, SystemState, WorkloadClassifier, current_settings, from_history, from_snapshot, plan,
)
from kernelsense.monitoring.classify import classify, WorkloadSignal
from kernelsense.ui.common import INK, SERIES, concept, get_recorder, page_setup, show, sidebar_status, style

page_setup("AI-Based Resource Allocation", "🤖")
rec = get_recorder()
sidebar_status(rec)

concept(
    "A Random Forest classifies each process from <b>how it uses the OS</b> - CPU demand, voluntary vs involuntary "
    "context switches, memory growth and I/O rate - into CPU-bound, I/O-bound, Interactive, Memory-heavy or Idle. "
    "A policy then maps each class to Linux levers: <b>nice</b> (CFS weight), <b>CPU affinity</b>, <b>I/O priority</b> "
    "and advisory scheduling-class / memory-limit changes. Actions are <b>dry-run by default</b> and reversible."
)


@st.cache_resource(show_spinner="Training workload classifier...")
def get_classifier() -> WorkloadClassifier:
    return WorkloadClassifier.load_or_train()


clf = get_classifier()

with st.sidebar:
    st.subheader("Policy")
    batch_nice = st.slider("Nice for CPU-bound background work", 1, 19, 10)
    io_level = st.slider("ionice level for I/O-bound work", 0, 7, 7)
    cpu_busy = st.slider("CPU busy threshold (%)", 10, 100, 70)
    min_conf = st.slider("Minimum classifier confidence", 0.3, 0.95, 0.6)
    feature_src = st.radio("Features from", ["Recorded history", "Latest snapshot"])

cfg = PolicyConfig(batch_nice=batch_nice, io_bound_ionice_level=io_level, cpu_busy_pct=cpu_busy,
                   min_confidence=min_conf)

tab_live, tab_model, tab_log, tab_rl, tab_loop = st.tabs(
    ["Classify & allocate", "Model", "Action log", "Policy learning (RL)", "Closed-loop control & cgroups"])

with tab_live:
    s, procs = rec.snapshot()
    if s is None:
        st.info("Waiting for data...")
        st.stop()
    feats = pd.DataFrame()
    if feature_src == "Recorded history":
        feats = from_history(rec.store.process_history(minutes=5))
    if feats.empty:
        active = [p for p in procs if p.cpu_percent > 0.2 or p.rss_mb > 150 or p.io_rate > 0 or p.vol_ctx_rate > 1]
        feats = from_snapshot(active)
        if feature_src == "Recorded history":
            st.caption("Not enough recorded history yet - using the latest snapshot.")
    classified = clf.predict(feats)
    if classified.empty:
        st.info("No active processes to classify.")
        st.stop()
    actions = plan(classified, SystemState(s.cpu_percent, s.mem_percent), cfg, current_settings(classified["pid"]))

    c = st.columns(4)
    c[0].metric("System CPU", f"{s.cpu_percent:.0f}%")
    c[1].metric("System memory", f"{s.mem_percent:.0f}%")
    c[2].metric("Processes classified", len(classified))
    c[3].metric(f"Planned actions (+{sum(not a.applicable for a in actions)} advisory)",
                sum(a.applicable for a in actions))

    a, b = st.columns([3, 2])
    with a:
        view = classified.sort_values(["cpu_mean", "rss_mb"], ascending=False)
        st.dataframe(view[["pid", "name", "predicted", "confidence", "cpu_mean", "rss_mb", "vol_rate", "invol_rate",
                           "io_rate_kb", "threads"]].round(2), hide_index=True, width="stretch", height=360,
                     column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1)})
    with b:
        counts = classified["predicted"].value_counts()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                               marker=dict(color=SERIES[0], cornerradius=4), text=counts.values,
                               textposition="outside", textfont=dict(color=INK["secondary"]),
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 360, "Predicted workload classes"), "classes")

    st.markdown("#### Planned actions")
    if not actions:
        st.success("No action needed: no CPU-bound work is competing with interactive tasks and memory is not under "
                   "pressure. (Run `python workloads/cpu_hog.py` and lower the CPU busy threshold to see actions.)")
    else:
        adf = pd.DataFrame([{"pid": x.pid, "name": x.name, "action": x.kind, "current": str(x.current),
                             "target": str(x.target), "type": "apply" if x.applicable else "advisory",
                             "reason": x.reason} for x in actions])
        st.dataframe(adf, hide_index=True, width="stretch")
        targets = sorted({x.pid for x in actions if x.applicable})
        c1, c2, c3 = st.columns([2, 1, 1])
        allow = c1.multiselect("Allow-list (only these PIDs may be changed)", targets, default=targets,
                               format_func=lambda p: f"{p} - {classified.set_index('pid').loc[p, 'name']}")
        dry = c2.toggle("Dry run", value=True, help="Simulate only. Turn off to really apply nice/affinity/ionice.")
        if c3.button("Execute plan", type="primary", width="stretch"):
            act = Actuator(dry_run=dry, allow_pids=set(allow))
            results = act.execute_all([x for x in actions if x.pid in set(allow) or not x.applicable])
            st.session_state.last_results = results
            for r in results:
                icon = {"applied": "✅", "dry-run": "🧪", "refused": "⛔", "failed": "❌", "advisory": "💡"}[r.status]
                st.write(f"{icon} **{r.status}** · {r.name} ({r.pid}) · {r.kind}: {r.message}")
        if any(r.status == "applied" for r in st.session_state.get("last_results", [])):
            if st.button("Roll back last applied actions"):
                undo = Actuator(dry_run=False).rollback(st.session_state.last_results)
                for r in undo:
                    st.write(f"↩️ {r.status} · {r.name} ({r.pid}) · {r.kind}: {r.message}")
                st.session_state.last_results = []
        st.caption("Raising nice is allowed for your own processes; lowering it (and rolling back a nice change) needs "
                   "root. Processes owned by other users and system processes are always refused.")

with tab_model:
    rep = clf.report
    if rep is None:
        rep = clf.train()
    c = st.columns(3)
    c[0].metric("Cross-validated accuracy", f"{rep.accuracy:.1%}")
    c[1].metric("Training samples", rep.n_samples)
    c[2].metric("Model", "Random Forest (200 trees)")
    a, b = st.columns(2)
    with a:
        cm = rep.confusion
        fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, text=cm.values, texttemplate="%{text}",
                                   colorscale=[[0, "#f0efec"], [0.5, "#86b6ef"], [1, "#184f95"]], showscale=False,
                                   hovertemplate="true %{y} → predicted %{x}: %{z}<extra></extra>",
                                   xgap=2, ygap=2))
        fig.update_layout(xaxis_title="predicted", yaxis_title="actual")
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 380, "Confusion matrix (5-fold CV)"), "cm")
    with b:
        imp = rep.importances
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 380, "Feature importance"), "imp")
    with st.expander("Classification report"):
        st.code(rep.report, language=None)
    st.caption("The model is trained on an OS-informed synthetic dataset (each class generated from the scheduler and "
               "memory behaviour that defines it). Retraining can add live processes labelled by the rule-based "
               "classifier as weak supervision.")
    if st.button("Retrain with live processes (weak labels)"):
        _, procs = rec.snapshot()
        live = from_snapshot(procs)
        rule_labels = [classify(WorkloadSignal(p.cpu_percent, int(p.vol_ctx_rate), int(p.invol_ctx_rate),
                                               int(p.io_rate), 1.0)) for p in procs]
        mapping = {"CPU-bound": "CPU-bound", "I/O-bound": "I/O-bound", "Interactive": "Interactive", "Idle": "Idle"}
        live["label"] = [mapping.get(l) for l in rule_labels]
        live = live.dropna(subset=["label"])
        new = WorkloadClassifier()
        new.train(extra=live)
        new.save()
        get_classifier.clear()
        st.success(f"Retrained with {len(live)} live samples. Accuracy {new.report.accuracy:.1%}.")
        st.rerun()

with tab_log:
    log = Actuator().read_log()
    if not log:
        st.info("No actions recorded yet.")
    else:
        ldf = pd.DataFrame(log)
        ldf["time"] = pd.to_datetime(ldf["ts"], unit="s")
        ldf[["previous", "target"]] = ldf[["previous", "target"]].astype(str)
        st.dataframe(ldf[["time", "status", "pid", "name", "kind", "previous", "target", "message"]].iloc[::-1],
                     hide_index=True, width="stretch")

with tab_rl:
    from kernelsense.allocator import env as sim_env
    from kernelsense.allocator.learning import evaluate as eval_policies, train as train_policies

    sim_env.load_calibration()
    st.markdown("Policies are trained in a **tick-based CFS contention simulator**: one latency-sensitive task, batch "
                "hogs whose nice value the agent controls, and other services it must not touch. Reward = batch "
                "throughput relative to its fair share minus a penalty when the interactive task misses its latency "
                f"SLO. The simulator's wake-up preemption is calibrated on this machine (ratio {sim_env.PREEMPT_RATIO}).")

    @st.cache_resource(show_spinner="Training LinUCB and Q-learning in the simulator...")
    def trained(episodes: int):
        bandit, q, hist = train_policies(episodes)
        return bandit, q, hist, eval_policies(bandit, q, 200)

    episodes = st.select_slider("Training episodes", [500, 1000, 2000, 3000], 2000)
    bandit, q, hist, table = trained(episodes)
    st.dataframe(table, hide_index=True, width="stretch")
    curve = hist.rolling(100).mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["episode"], y=curve["q_reward"], name="Q-learning", line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=curve["episode"], y=curve["bandit_reward"], name="LinUCB", line=dict(color=SERIES[1], width=2)))
    fig.update_layout(xaxis_title="episode", yaxis_title="reward (100-episode mean)", hovermode="x unified")
    show(style(fig, 300, "Learning curves"), "rlcurve")
    st.markdown("**Learned Q-table** (best batch nice per workload state)")
    st.dataframe(q.policy_table(), hide_index=True, width="stretch")
    if st.button("Save Q-policy for the controller"):
        from kernelsense.config import settings
        st.success(f"Saved to {q.save(settings.data_dir / 'q_policy.json')}")

with tab_loop:
    from kernelsense.allocator.cgroup import detect as detect_cg
    from kernelsense.allocator.controller import ClosedLoopController, ControllerConfig

    info = detect_cg()
    c = st.columns(3)
    c[0].metric("cgroup version", f"v{info.version}" if info else "none")
    c[1].metric("Writable", "yes" if info and info.writable else "no")
    c[2].metric("Controllers", ", ".join(info.controllers) if info else "-")
    st.markdown("The controller runs **observe → decide → act → measure**: it reads per-CPU contention from /proc, picks "
                "a nice level for the selected batch processes from the learned policy, applies it via `nice` or a "
                "cgroup's CPU weight, and rolls back if CPU pressure (PSI) gets worse.")
    _, procs = rec.snapshot()
    choices = sorted([p for p in procs if p.cpu_percent > 5], key=lambda p: -p.cpu_percent)
    label = {p.pid: f"{p.pid} - {p.name} ({p.cpu_percent:.0f}% CPU)" for p in procs}
    batch = st.multiselect("Batch processes to manage", [p.pid for p in choices], format_func=lambda x: label.get(x, x))
    inter = st.multiselect("Latency-sensitive processes", [p.pid for p in procs if p.pid not in batch],
                           format_func=lambda x: label.get(x, x))
    c1, c2, c3 = st.columns(3)
    pol = c1.selectbox("Policy", ["q", "bandit", "rule", "none"])
    mode = c2.selectbox("Actuation", ["nice", "cgroup"], disabled=not (info and info.writable))
    steps = c3.number_input("Control steps (1 s each)", 1, 30, 5)
    confirm = st.checkbox("I understand this changes the priority of the selected processes (restored afterwards)")
    if st.button("Run controller", type="primary", disabled=not (batch and confirm)):
        try:
            ctl = ClosedLoopController(batch, inter, ControllerConfig(policy=pol, mode=mode, interval=1.0))
            rows = []
            with st.spinner("Controlling..."):
                for _ in range(int(steps)):
                    d = ctl.step()
                    rows.append({"state": str(d.state), "nice": d.nice, "applied": d.applied,
                                 "psi_before": d.psi_before, "psi_after": d.psi_after, "rolled_back": d.rolled_back,
                                 "note": d.note})
            ctl.close()
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.success("Done - settings restored.")
        except (RuntimeError, OSError) as exc:
            st.error(str(exc))
    st.caption("Measured effect on this machine is reported on the Benchmarks page (`python benchmarks/run_all.py allocation`).")
