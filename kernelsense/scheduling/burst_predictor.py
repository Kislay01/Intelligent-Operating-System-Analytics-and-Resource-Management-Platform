"""Next CPU-burst prediction by exponential averaging.

    tau(n+1) = alpha * t(n) + (1 - alpha) * tau(n)

t(n)   : length of the n-th observed burst
tau(n) : predicted length of the n-th burst
alpha  : weight of recent history (0 = ignore history, 1 = only last burst)

This is how SJF/SRTF can be approximated in practice, since the true next burst
is unknown to the OS.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionTrace:
    actual: list[float]
    predicted: list[float]  # predicted[i] is the prediction made *before* actual[i]
    alpha: float

    @property
    def mae(self) -> float:
        if not self.actual:
            return 0.0
        return sum(abs(a - p) for a, p in zip(self.actual, self.predicted)) / len(self.actual)

    @property
    def next_prediction(self) -> float:
        if not self.actual:
            return self.predicted[0] if self.predicted else 0.0
        return self.alpha * self.actual[-1] + (1 - self.alpha) * self.predicted[-1]


def exponential_average(bursts: list[float], alpha: float = 0.5, tau0: float | None = None) -> PredictionTrace:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not bursts:
        return PredictionTrace([], [tau0 or 0.0], alpha)
    tau = float(tau0 if tau0 is not None else bursts[0])
    preds = []
    for b in bursts:
        preds.append(tau)
        tau = alpha * b + (1 - alpha) * tau
    return PredictionTrace(list(map(float, bursts)), preds, alpha)


def best_alpha(bursts: list[float], grid: int = 21) -> tuple[float, float]:
    """Grid-search alpha that minimises MAE on the observed history."""
    best = (0.5, float("inf"))
    for i in range(grid):
        a = i / (grid - 1)
        m = exponential_average(bursts, a).mae
        if m < best[1]:
            best = (a, m)
    return best
