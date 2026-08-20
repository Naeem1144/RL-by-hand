"""Epsilon-greedy bandit algorithm."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class EpsilonGreedy(BanditAlgorithm):
    """Explore randomly with probability epsilon, otherwise exploit.

    Values are sample means by default.  Pass ``step_size`` to use an
    exponential recency-weighted average instead, which tracks non-stationary
    problems.  Pass ``optimistic_value`` above every arm's mean to encourage
    early exploration through pessimistic initial estimates.
    """

    epsilon: float = 0.1
    decay_rate: float | None = None
    step_size: float | None = None
    optimistic_value: float | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)
    _current_epsilon: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.epsilon = float(self.epsilon)
        if not np.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be finite and lie in [0, 1]")
        if self.decay_rate is not None:
            self.decay_rate = float(self.decay_rate)
            if not np.isfinite(self.decay_rate) or not 0.0 < self.decay_rate < 1.0:
                raise ValueError("decay_rate must lie strictly between 0 and 1")
        if self.step_size is not None:
            self.step_size = float(self.step_size)
            if not np.isfinite(self.step_size) or not 0.0 < self.step_size <= 1.0:
                raise ValueError("step_size must be finite and lie in (0, 1]")
        if self.optimistic_value is not None:
            self.optimistic_value = float(self.optimistic_value)
            if not np.isfinite(self.optimistic_value):
                raise ValueError("optimistic_value must be finite")

    @property
    def name(self) -> str:
        suffix = f", decay={self.decay_rate:g}" if self.decay_rate is not None else ""
        suffix += f", alpha={self.step_size:g}" if self.step_size is not None else ""
        suffix += ", optimistic" if self.optimistic_value is not None else ""
        return f"Epsilon-greedy(epsilon={self.epsilon:g}{suffix})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        self._rng = rng
        initial_value = 0.0 if self.optimistic_value is None else self.optimistic_value
        self._values = np.full(n_arms, initial_value)
        self._pulls = np.zeros(n_arms, dtype=int)
        self._current_epsilon = self.epsilon

    def select_arm(self, step: int) -> int:
        del step
        if self._rng.random() < self._current_epsilon:
            return int(self._rng.integers(self._values.size))
        best = np.flatnonzero(self._values == self._values.max())
        return int(best[0]) if best.size == 1 else int(self._rng.choice(best))

    def update(self, arm: int, reward: float) -> None:
        self._pulls[arm] += 1
        if self.step_size is None:
            self._values[arm] += (reward - self._values[arm]) / self._pulls[arm]
        else:
            self._values[arm] += self.step_size * (reward - self._values[arm])
        if self.decay_rate is not None:
            self._current_epsilon *= self.decay_rate

    @property
    def estimated_values(self) -> np.ndarray:
        return self._values

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
