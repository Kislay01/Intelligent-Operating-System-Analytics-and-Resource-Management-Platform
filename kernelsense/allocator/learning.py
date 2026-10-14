"""Learning allocation policies in the CPU contention simulator.

* LinUCB contextual bandit (Li et al., WWW 2010): one ridge-regression model of
  reward per action over the workload context; picks the action with the
  highest upper confidence bound, balancing exploration and exploitation.
* Tabular Q-learning: discretised workload state x batch-nice action. Each
  scenario is a one-step episode, so the update reduces to an incremental
  average of observed rewards with epsilon-greedy exploration.

Both are compared with fixed baselines on held-out scenarios.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .env import ACTIONS, Scenario, context_of, random_scenario, simulate, state_of


class LinUCB:
    def __init__(self, n_actions: int = len(ACTIONS), dim: int = 6, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self.A = [np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]

    def choose(self, x: np.ndarray, explore: bool = True) -> int:
        scores = []
        for A, b in zip(self.A, self.b):
            inv = np.linalg.inv(A)
            theta = inv @ b
            bonus = self.alpha * np.sqrt(x @ inv @ x) if explore else 0.0
            scores.append(theta @ x + bonus)
        return int(np.argmax(scores))

    def update(self, a: int, x: np.ndarray, reward: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += reward * x


@dataclass
class QLearner:
    epsilon: float = 0.2
    lr: float = 0.1
    q: dict = field(default_factory=dict)

    def _row(self, s) -> list[float]:
        return self.q.setdefault(tuple(s), [0.0] * len(ACTIONS))

    def choose(self, s, explore: bool = True, rng: random.Random | None = None) -> int:
        rng = rng or random
        if explore and rng.random() < self.epsilon:
            return rng.randrange(len(ACTIONS))
        row = self._row(s)
        return int(np.argmax(row))

    def update(self, s, a: int, reward: float) -> None:
        row = self._row(s)
        row[a] += self.lr * (reward - row[a])  # one-step episodes: target = reward

    def policy_table(self) -> pd.DataFrame:
        rows = []
        for s, vals in sorted(self.q.items()):
            rows.append({"cpu_hogs": ["0", "1-2", "3-4", "5+"][s[0]], "interactive_load": ["<10%", "10-25%", ">25%"][s[1]],
                         "batch_fraction": ["none", "<60%", ">=60%"][s[2]],
                         "best_nice": ACTIONS[int(np.argmax(vals))], **{f"Q(nice={a})": round(v, 3)
                                                                         for a, v in zip(ACTIONS, vals)}})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"epsilon": self.epsilon, "lr": self.lr,
                                    "q": {",".join(map(str, k)): v for k, v in self.q.items()}}))
        return path

    @staticmethod
    def load(path: Path) -> "QLearner":
        d = json.loads(Path(path).read_text())
        q = QLearner(d["epsilon"], d["lr"])
        q.q = {tuple(int(x) for x in k.split(",")): v for k, v in d["q"].items()}
        return q


def train(episodes: int = 3000, seed: int = 0, lam: float = 0.6, horizon: int = 1000):
    rng = random.Random(seed)
    bandit, qlearn = LinUCB(), QLearner()
    history = []
    for ep in range(episodes):
        sc = random_scenario(rng)
        x = context_of(sc)
        a_b = bandit.choose(x)
        r_b = simulate(sc, ACTIONS[a_b], horizon, lam, seed=ep).reward
        bandit.update(a_b, x, r_b)
        s = state_of(sc)
        a_q = qlearn.choose(s, rng=rng)
        r_q = simulate(sc, ACTIONS[a_q], horizon, lam, seed=ep).reward
        qlearn.update(s, a_q, r_q)
        history.append({"episode": ep, "bandit_reward": r_b, "q_reward": r_q})
    return bandit, qlearn, pd.DataFrame(history)


def rule_policy(sc: Scenario) -> int:
    """Hand-written baseline: de-prioritise batch work when there is an interactive task competing."""
    return 10 if sc.n_batch + sc.n_other >= 2 else 0


def evaluate(bandit: LinUCB, qlearn: QLearner, n: int = 300, seed: int = 99, lam: float = 0.6,
             horizon: int = 1000) -> pd.DataFrame:
    rng = random.Random(seed)
    policies = {
        "default (nice 0)": lambda sc: 0,
        "always nice 19": lambda sc: 19,
        "rule-based": rule_policy,
        "LinUCB bandit": lambda sc: ACTIONS[bandit.choose(context_of(sc), explore=False)],
        "Q-learning": lambda sc: ACTIONS[qlearn.choose(state_of(sc), explore=False)],
    }
    rows = []
    for i in range(n):
        sc = random_scenario(rng)
        best = max((simulate(sc, a, horizon, lam, seed=10_000 + i).reward, a) for a in ACTIONS)
        for name, pol in policies.items():
            a = pol(sc)
            res = simulate(sc, a, horizon, lam, seed=10_000 + i)
            rows.append({"policy": name, "scenario": i, "nice": a, "reward": res.reward,
                         "latency_p95": res.latency_p95, "batch_share": res.batch_share,
                         "regret": best[0] - res.reward})
    df = pd.DataFrame(rows)
    return df.groupby("policy").agg(
        reward=("reward", "mean"), latency_p95=("latency_p95", "mean"), batch_share=("batch_share", "mean"),
        regret=("regret", "mean"),
    ).sort_values("reward", ascending=False).round(4).reset_index()
