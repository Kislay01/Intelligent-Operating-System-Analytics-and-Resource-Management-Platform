"""DeepLog-style sequence anomaly detection over log templates.

DeepLog (Du et al., CCS 2017) models a system's normal execution as the
sequence of log keys (templates) it emits, learns to predict the next key from
the previous h keys, and flags a key as anomalous when it is not among the top
k predictions. DeepLog uses an LSTM; this implementation uses the same
detection rule with an n-gram (order h) model with back-off, which needs no
deep-learning dependency and trains in milliseconds.
"""

from __future__ import annotations

from collections import Counter, defaultdict


class NGramSequenceModel:
    def __init__(self, history: int = 3, top_k: int = 5) -> None:
        self.h = history
        self.k = top_k
        self.tables: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(history + 1)]
        self.vocab: Counter = Counter()

    def fit(self, sequences: list[list[int]]) -> "NGramSequenceModel":
        for seq in sequences:
            for i, key in enumerate(seq):
                self.vocab[key] += 1
                for order in range(self.h + 1):
                    if i - order < 0:
                        break
                    ctx = tuple(seq[i - order : i])
                    self.tables[order][ctx][key] += 1
        return self

    def top_candidates(self, context: list[int]) -> list[int]:
        """Back off from the longest context that has been seen."""
        for order in range(min(self.h, len(context)), -1, -1):
            ctx = tuple(context[len(context) - order :]) if order else ()
            counts = self.tables[order].get(ctx)
            if counts:
                return [k for k, _ in counts.most_common(self.k)]
        return []

    def anomalies(self, seq: list[int]) -> list[bool]:
        """Per-position flag: True if the key was not in the top-k predictions."""
        flags = []
        for i, key in enumerate(seq):
            if key not in self.vocab:
                flags.append(True)
                continue
            flags.append(i >= 1 and key not in self.top_candidates(seq[max(0, i - self.h) : i]))
        return flags

    def is_anomalous(self, seq: list[int], min_flags: int = 1) -> bool:
        return sum(self.anomalies(seq)) >= min_flags
