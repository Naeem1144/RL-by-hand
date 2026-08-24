"""Interface for stationary multi-armed bandit problems."""

from typing import Protocol

import numpy as np


class BanditProblem(Protocol):
    """A reward source with an expected reward for every arm.

    Problems may be stateful: ``sample`` is allowed to advance internal
    state, as the drifting Bernoulli bandit does.  A stateful problem must
    set ``mutating = True`` so that ``evaluation.compare`` can refuse to
    share one instance across runs and ask for a factory instead of
    silently corrupting results.  Stateless problems need not declare it.
    """

    mutating: bool = False

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
