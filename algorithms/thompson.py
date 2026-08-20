"""Beta-Bernoulli Thompson sampling bandit algorithm."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class ThompsonSampling(BanditAlgorithm):
    """Sample from one Beta posterior per Bernoulli arm."""

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _successes: np.ndarray = field(init=False, repr=False)
    _failures: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.prior_alpha = float(self.prior_alpha)
        self.prior_beta = float(self.prior_beta)
        if (
            not np.isfinite(self.prior_alpha)
            or not np.isfinite(self.prior_beta)
            or self.prior_alpha <= 0
            or self.prior_beta <= 0
        ):
            raise ValueError("prior_alpha and prior_beta must be positive")

    @property
    def name(self) -> str:
        return f"Thompson(alpha={self.prior_alpha:g}, beta={self.prior_beta:g})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        self._rng = rng
        self._successes = np.zeros(n_arms, dtype=int)
        self._failures = np.zeros(n_arms, dtype=int)
        self._pulls = np.zeros(n_arms, dtype=int)

    def select_arm(self, step: int) -> int:
        del step
        samples = self._rng.beta(
            self._successes + self.prior_alpha,
            self._failures + self.prior_beta,
        )
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if reward not in (0.0, 1.0):
            raise ValueError("ThompsonSampling requires binary rewards")
        self._successes[arm] += int(reward)
        self._failures[arm] += int(1 - reward)
        self._pulls[arm] += 1

    @property
    def estimated_values(self) -> np.ndarray:
        return (self._successes + self.prior_alpha) / (
            self._pulls + self.prior_alpha + self.prior_beta
        )

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
