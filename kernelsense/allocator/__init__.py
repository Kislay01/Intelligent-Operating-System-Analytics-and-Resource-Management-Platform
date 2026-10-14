"""AI-based resource allocation: features, workload classifier, policy and actuator."""

from __future__ import annotations

import psutil

from .actuator import ActionResult, Actuator
from .cgroup import Cgroup, CgroupInfo, detect as detect_cgroups
from .controller import ClosedLoopController, ControllerConfig, Decision
from .env import ACTIONS, Scenario, simulate
from .learning import LinUCB, QLearner, evaluate as evaluate_policies, train as train_policies
from .classifier import LABELS, TrainingReport, WorkloadClassifier, synthetic_dataset
from .features import FEATURES, from_history, from_snapshot
from .policy import Action, PolicyConfig, SystemState, plan


def current_settings(pids) -> dict:
    out = {}
    for pid in pids:
        try:
            p = psutil.Process(int(pid))
            out[int(pid)] = {
                "nice": p.nice(),
                "affinity": p.cpu_affinity() if hasattr(p, "cpu_affinity") else [],
            }
        except (psutil.Error, OSError):
            continue
    return out


def recommend(features, system: SystemState, classifier: WorkloadClassifier | None = None,
              cfg: PolicyConfig | None = None):
    """Classify processes and return (classified DataFrame, planned actions)."""
    clf = classifier or WorkloadClassifier.load_or_train()
    classified = clf.predict(features)
    actions = plan(classified, system, cfg, current_settings(classified["pid"]) if not classified.empty else {})
    return classified, actions


__all__ = [
    "ACTIONS", "Cgroup", "CgroupInfo", "ClosedLoopController", "ControllerConfig", "Decision", "LinUCB", "QLearner",
    "Scenario", "detect_cgroups", "evaluate_policies", "simulate", "train_policies",
    "Action", "ActionResult", "Actuator", "FEATURES", "LABELS", "PolicyConfig", "SystemState", "TrainingReport",
    "WorkloadClassifier", "current_settings", "from_history", "from_snapshot", "plan", "recommend",
    "synthetic_dataset",
]
