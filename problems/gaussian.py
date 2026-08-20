"""A stationary Gaussian multi-armed bandit problem."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class GaussianBandit:
    """A problem whose arm rewards are independent Gaussian draws.

    ``means[i]`` is the expected reward of arm ``i`` and every pull adds
    Gaussian noise with standard deviation ``reward_std``.  Unlike the
    Bernoulli problem this supports unbounded real-valued rewards.
    """

    means: np.ndarray
    reward_std: float = 1.0
    name: str = "Gaussian bandit"

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        if means.ndim != 1 or means.size == 0:
            raise ValueError("means must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(means)):
            raise ValueError("means must contain only finite values")

        reward_std = float(self.reward_std)
        if not np.isfinite(reward_std) or reward_std <= 0.0:
            raise ValueError("reward_std must be finite and positive")

        means = means.copy()
        means.setflags(write=False)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "reward_std", reward_std)

    @classmethod
    def random(cls, n_arms: int, seed: int = 0, name: str = "Random Gaussian bandit"):
        """Create a reproducible problem with Uniform(-1, 3) arm means."""
        if isinstance(n_arms, bool) or not isinstance(n_arms, Integral):
            raise TypeError("n_arms must be an integer")
        if int(n_arms) <= 0:
            raise ValueError("n_arms must be positive")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        means = np.random.default_rng(int(seed)).uniform(-1.0, 3.0, int(n_arms))
        return cls(means, name=name)

    @property
    def n_arms(self) -> int:
        return int(self.means.size)

    @property
    def expected_rewards(self) -> np.ndarray:
        return self.means

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        if isinstance(arm, bool) or not isinstance(arm, Integral):
            raise TypeError("arm must be an integer")
        arm = int(arm)
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must lie in [0, {self.n_arms})")
        return float(rng.normal(self.means[arm], self.reward_std))
