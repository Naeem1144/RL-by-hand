"""A non-stationary Bernoulli bandit whose arm means random-walk over time."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

PROBABILITY_BOUNDS = (0.02, 0.98)


@dataclass
class DriftingBernoulliBandit:
    """A Bernoulli bandit whose probabilities drift every step.

    Before each reward draw every arm's probability takes a Gaussian random
    -walk step of standard deviation ``drift_std`` and is clipped to
    ``PROBABILITY_BOUNDS``.  ``expected_rewards`` is a live view of the
    current arm means, so it changes between calls.

    Because the problem mutates, pass a factory to ``compare`` so every run
    receives its own instance; matched seeds then reproduce identical drift
    trajectories for every algorithm.  The ``mutating`` marker makes
    ``compare`` enforce this instead of silently sharing one instance.
    """

    mutating = True

    initial_probabilities: np.ndarray
    drift_std: float = 0.02
    name: str = "Drifting Bernoulli bandit"
    _probabilities: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.initial_probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError("initial_probabilities must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("initial_probabilities must contain only finite values")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("initial_probabilities must lie in [0, 1]")

        drift_std = float(self.drift_std)
        if not np.isfinite(drift_std) or drift_std <= 0.0:
            raise ValueError("drift_std must be finite and positive")

        self.initial_probabilities = probabilities.copy()
        self._probabilities = probabilities.copy()

    @property
    def n_arms(self) -> int:
        return int(self._probabilities.size)

    @property
    def expected_rewards(self) -> np.ndarray:
        return self._probabilities.copy()

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        if isinstance(arm, bool) or not isinstance(arm, Integral):
            raise TypeError("arm must be an integer")
        arm = int(arm)
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm must lie in [0, {self.n_arms})")

        steps = rng.normal(0.0, self.drift_std, self._probabilities.size)
        self._probabilities += steps
        np.clip(
            self._probabilities,
            PROBABILITY_BOUNDS[0],
            PROBABILITY_BOUNDS[1],
            out=self._probabilities,
        )
        return float(rng.random() < self._probabilities[arm])
