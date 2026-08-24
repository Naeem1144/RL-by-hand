"""A stationary Bernoulli multi-armed bandit problem."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class BernoulliBandit:
    """A problem whose arm rewards are independent Bernoulli draws.

    ``probabilities[i]`` is the probability of receiving reward 1 from arm
    ``i``.  Passing explicit probabilities makes it easy to run several
    algorithms on exactly the same problem.
    """

    probabilities: np.ndarray
    name: str = "Bernoulli bandit"

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError("probabilities must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must contain only finite values")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")

        probabilities = probabilities.copy()
        probabilities.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def random(cls, n_arms: int, seed: int = 0, name: str = "Random Bernoulli bandit"):
        """Create a reproducible problem with Uniform(0, 1) arm means."""
        if isinstance(n_arms, bool) or not isinstance(n_arms, Integral):
            raise TypeError("n_arms must be an integer")
        if int(n_arms) <= 0:
            raise ValueError("n_arms must be positive")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
        return cls(np.random.default_rng(int(seed)).random(int(n_arms)), name=name)

    @property
    def n_arms(self) -> int:
        return int(self.probabilities.size)

    @property
    def expected_rewards(self) -> np.ndarray:
        return self.probabilities

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        if isinstance(arm, bool) or not isinstance(arm, Integral):
            raise TypeError("arm must be an integer")
        arm = int(arm)
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must lie in [0, {self.n_arms})")
        return float(rng.random() < self.probabilities[arm])
