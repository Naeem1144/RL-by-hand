"""Upper confidence bound (UCB) bandit algorithm."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class UCB(BanditAlgorithm):
    """Select the largest empirical mean plus confidence bonus."""

    c: float = float(np.sqrt(2.0))
    _values: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.c = float(self.c)
        if not np.isfinite(self.c) or self.c < 0.0:
            raise ValueError("c must be finite and non-negative")

    @property
    def name(self) -> str:
        return f"UCB(c={self.c:g})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        del rng
        self._values = np.zeros(n_arms)
        self._pulls = np.zeros(n_arms, dtype=int)

    def select_arm(self, step: int) -> int:
        untried = np.flatnonzero(self._pulls == 0)
        if untried.size:
            return int(untried[0])
        scores = self._values + self.c * np.sqrt(np.log(step + 1) / self._pulls)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        self._pulls[arm] += 1
        self._values[arm] += (reward - self._values[arm]) / self._pulls[arm]

    @property
    def estimated_values(self) -> np.ndarray:
        return self._values

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
