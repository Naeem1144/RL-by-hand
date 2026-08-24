"""Variance-aware upper confidence bound (UCB-V) bandit algorithm."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class UCBVariance(BanditAlgorithm):
    """UCB with a variance-aware bonus (Audibert, Munos, Szepesvari, 2009).

    The index is ``mean + c * sqrt(variance * log(t) / n) + b * log(t) / n``
    with ``t`` the current time step.  The empirical variance enters the
    confidence term, so arms whose rewards are already well determined are
    explored less; for binary rewards the bonus vanishes as the mean
    approaches 0 or 1.  The additive ``b * log(t) / n`` term guards against
    un-pulled small-n arms with underestimated variance.

    The variance is maintained with Welford's online algorithm, so the
    estimate stays numerically stable over very long runs.
    """

    c: float = float(np.sqrt(2.0))
    b: float = 3.0
    _means: np.ndarray = field(init=False, repr=False)
    _m2: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.c = float(self.c)
        self.b = float(self.b)
        if not np.isfinite(self.c) or self.c < 0.0:
            raise ValueError("c must be finite and non-negative")
        if not np.isfinite(self.b) or self.b < 0.0:
            raise ValueError("b must be finite and non-negative")

    @property
    def name(self) -> str:
        return f"UCB-V(c={self.c:.5g}, b={self.b:.5g})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        del rng
        self._means = np.zeros(n_arms)
        self._m2 = np.zeros(n_arms)
        self._pulls = np.zeros(n_arms, dtype=int)

    def select_arm(self, step: int) -> int:
        untried = np.flatnonzero(self._pulls == 0)
        if untried.size:
            return int(untried[0])
        t = int(step) + 1
        log_t = np.log(t)
        variance = np.maximum(self._m2 / self._pulls, 0.0)
        scores = (
            self._means
            + self.c * np.sqrt(variance * log_t / self._pulls)
            + self.b * log_t / self._pulls
        )
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        self._pulls[arm] += 1
        delta = reward - self._means[arm]
        self._means[arm] += delta / self._pulls[arm]
        self._m2[arm] += delta * (reward - self._means[arm])

    @property
    def estimated_values(self) -> np.ndarray:
        return self._means

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
