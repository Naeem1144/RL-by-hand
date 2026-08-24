"""Beta-Bernoulli Thompson sampling with exponentially decaying counts."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class ForgettingThompsonSampling(BanditAlgorithm):
    """Thompson sampling with exponentially discounted posterior counts.

    At every step the pseudo-counts of every arm are multiplied by
    ``gamma`` before the new reward is added, so the effective horizon is
    about ``1 / (1 - gamma)`` observations.  This keeps the posterior
    responsive on drifting problems while retaining the Beta-Bernoulli
    conjugate update of :class:`algorithms.thompson.ThompsonSampling`
    (which is recovered exactly at ``gamma = 1``).
    """

    gamma: float = 0.99
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _successes: np.ndarray = field(init=False, repr=False)
    _failures: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.gamma = float(self.gamma)
        self.prior_alpha = float(self.prior_alpha)
        self.prior_beta = float(self.prior_beta)
        if not np.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be finite and lie in (0, 1]")
        if (
            not np.isfinite(self.prior_alpha)
            or not np.isfinite(self.prior_beta)
            or self.prior_alpha <= 0
            or self.prior_beta <= 0
        ):
            raise ValueError("prior_alpha and prior_beta must be positive")

    @property
    def name(self) -> str:
        return f"Thompson decay={self.gamma:g}"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        self._rng = rng
        self._successes = np.zeros(n_arms, dtype=float)
        self._failures = np.zeros(n_arms, dtype=float)
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
            raise ValueError("ForgettingThompsonSampling requires binary rewards")
        self._successes *= self.gamma
        self._failures *= self.gamma
        self._successes[arm] += reward
        self._failures[arm] += int(1 - reward)
        self._pulls[arm] += 1

    @property
    def estimated_values(self) -> np.ndarray:
        return (self._successes + self.prior_alpha) / (
            self._successes + self._failures + self.prior_alpha + self.prior_beta
        )

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
