"""Interface for stationary multi-armed bandit problems."""

from typing import Protocol

import numpy as np


class BanditProblem(Protocol):
    """A reward source with a fixed expected reward for every arm."""

    @property
    def name(self) -> str:
        """Human-readable problem name."""
        ...

    @property
    def n_arms(self) -> int:
        """Number of actions available to an algorithm."""
        ...

    @property
    def expected_rewards(self) -> np.ndarray:
        """Expected reward of every arm."""
        ...

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        """Draw one reward for ``arm`` using ``rng``."""
        ...
