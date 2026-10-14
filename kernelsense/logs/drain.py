"""Drain: online log template mining (He et al., ICWS 2017).

Log lines are produced by a small number of print statements with variable
parameters ("Accepted password for alice from 10.0.0.3"). Drain recovers those
statements as *templates* ("Accepted password for <*> from <*>") in one pass
using a fixed-depth parse tree:

    root -> token count -> first ``depth - 2`` tokens -> list of log groups

A new line descends the tree and joins the most similar group if the share of
identical tokens is >= ``sim_threshold``; differing positions become ``<*>``.
Otherwise it starts a new group. Common variables (IPs, numbers, hex, paths)
are masked first so they do not split groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"(?<![\w/])/[\w.\-/]+"), "<PATH>"),
    (re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])"), "<NUM>"),
]
WILDCARD = "<*>"


@dataclass
class LogCluster:
    cluster_id: int
    template: list[str]
    size: int = 0
    members: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.template)


def preprocess(line: str) -> list[str]:
    for pattern, token in MASKS:
        line = pattern.sub(token, line)
    return line.strip().split()


class Drain:
    def __init__(self, depth: int = 4, sim_threshold: float = 0.5, max_children: int = 100,
                 keep_members: bool = True) -> None:
        if depth < 3:
            raise ValueError("depth must be >= 3")
        self.depth = depth
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.keep_members = keep_members
        self.root: dict = {}
        self.clusters: list[LogCluster] = []
        self._n = 0

    @staticmethod
    def _has_digit(tok: str) -> bool:
        return any(ch.isdigit() for ch in tok) or tok.startswith("<")

    def _leaf(self, tokens: list[str], create: bool) -> list[LogCluster] | None:
        node = self.root.setdefault(len(tokens), {}) if create else self.root.get(len(tokens))
        if node is None:
            return None
        for tok in tokens[: self.depth - 2]:
            key = WILDCARD if self._has_digit(tok) else tok
            if key not in node:
                if not create:
                    key = WILDCARD if WILDCARD in node else None
                    if key is None:
                        return None
                elif len(node) >= self.max_children:
                    key = WILDCARD
            node = node.setdefault(key, {}) if create else node[key]
        if create:
            return node.setdefault("__clusters__", [])
        return node.get("__clusters__")

    @staticmethod
    def _similarity(template: list[str], tokens: list[str]) -> tuple[float, int]:
        same = params = 0
        for a, b in zip(template, tokens):
            if a == WILDCARD:
                params += 1
            elif a == b:
                same += 1
        return same / len(tokens), params

    def add(self, line: str) -> LogCluster:
        tokens = preprocess(line) or ["<EMPTY>"]
        leaf = self._leaf(tokens, create=True)
        best, best_sim, best_params = None, -1.0, -1
        for c in leaf:
            sim, params = self._similarity(c.template, tokens)
            if sim > best_sim or (sim == best_sim and params > best_params):
                best, best_sim, best_params = c, sim, params
        if best is not None and best_sim >= self.sim_threshold:
            best.template = [a if a == b else WILDCARD for a, b in zip(best.template, tokens)]
            cluster = best
        else:
            cluster = LogCluster(len(self.clusters) + 1, tokens)
            self.clusters.append(cluster)
            leaf.append(cluster)
        cluster.size += 1
        if self.keep_members:
            cluster.members.append(self._n)
        self._n += 1
        return cluster

    def fit(self, lines) -> list[int]:
        """Parse many lines; returns the cluster id assigned to each line (at the time it was seen)."""
        return [self.add(l).cluster_id for l in lines]

    def match(self, line: str) -> LogCluster | None:
        """Find the template for a line without changing the model."""
        tokens = preprocess(line) or ["<EMPTY>"]
        leaf = self._leaf(tokens, create=False) or []
        best = max(leaf, key=lambda c: self._similarity(c.template, tokens)[0], default=None)
        if best is not None and self._similarity(best.template, tokens)[0] >= self.sim_threshold:
            return best
        return None

    def final_assignments(self) -> list[int]:
        """Cluster id of every parsed line according to the final clusters."""
        out = [0] * self._n
        for c in self.clusters:
            for m in c.members:
                out[m] = c.cluster_id
        return out

    def templates(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"cluster_id": c.cluster_id, "size": c.size, "template": c.text} for c in self.clusters]
        ).sort_values("size", ascending=False).reset_index(drop=True)


def grouping_accuracy(predicted: list, truth: list) -> float:
    """Loghub 'parsing accuracy': share of lines whose predicted group contains exactly
    the same set of lines as their ground-truth event group."""
    df = pd.DataFrame({"pred": predicted, "truth": truth})
    pred_groups = df.groupby("pred").groups
    truth_groups = df.groupby("truth").groups
    correct = 0
    for _, idx in pred_groups.items():
        truths = df.loc[idx, "truth"].unique()
        if len(truths) == 1 and len(truth_groups[truths[0]]) == len(idx):
            correct += len(idx)
    return correct / len(df) if len(df) else 0.0
