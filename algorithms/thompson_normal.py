"""Normal Thompson sampling bandit algorithm for real-valued rewards."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class NormalThompsonSampling(BanditAlgorithm):
    """Sample from one normal posterior per arm with known reward noise.

    The reward of every arm is modelled as Gaussian with known standard
    deviation ``reward_std`` and an unknown mean given a normal prior.  The
    posterior is conjugate, so each update is a precision-weighted average
    of the prior mean and the observed rewards.
    """

    reward_std: float = 1.0
    prior_mean: float = 0.0
    prior_std: float = 1.0
    _rng: np.random.Generator = field(init=False, repr=False)
    _sums: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reward_std = float(self.reward_std)
        self.prior_mean = float(self.prior_mean)
        self.prior_std = float(self.prior_std)
        if not np.isfinite(self.reward_std) or self.reward_std <= 0:
            raise ValueError("reward_std must be finite and positive")
        if not np.isfinite(self.prior_mean):
            raise ValueError("prior_mean must be finite")
        if not np.isfinite(self.prior_std) or self.prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")

    @property
    def name(self) -> str:
        return (
            f"Thompson-Normal(mu0={self.prior_mean:g}, "
            f"sigma0={self.prior_std:g}, sigma={self.reward_std:g})"
        )

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        self._rng = rng
        self._sums = np.zeros(n_arms)
        self._pulls = np.zeros(n_arms, dtype=int)

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        prior_precision = 1.0 / self.prior_std**2
        noise_precision = 1.0 / self.reward_std**2
        precision = prior_precision + self._pulls * noise_precision
        mean = (self.prior_mean * prior_precision + self._sums * noise_precision) / precision
        return mean, precision

    def select_arm(self, step: int) -> int:
        del step
        mean, precision = self._posterior()
        samples = self._rng.normal(mean, np.sqrt(1.0 / precision))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        self._sums[arm] += reward
        self._pulls[arm] += 1

    @property
    def estimated_values(self) -> np.ndarray:
        mean, _ = self._posterior()
        return mean

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
