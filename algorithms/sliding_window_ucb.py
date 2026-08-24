"""Sliding-window upper confidence bound bandit algorithm."""

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from algorithms.base import BanditAlgorithm


@dataclass
class SlidingWindowUCB(BanditAlgorithm):
    """UCB on the last ``window`` rewards (Garivier & Moulines, 2011).

    Only the most recent ``window`` observations per arm contribute to the
    empirical mean and the confidence bonus, so outdated rewards age out.
    The bonus uses ``log(min(t, window))``, matching the sliding-window
    analysis; all-time pull counts are still reported through
    :attr:`total_pulls` so the shared runner contract holds.
    """

    window: int = 1_000
    c: float = float(np.sqrt(2.0))
    _sums: np.ndarray = field(init=False, repr=False)
    _rewards: list = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(self.window, Integral):
            raise TypeError("window must be an integer")
        self.window = int(self.window)
        if self.window <= 0:
            raise ValueError("window must be positive")
        self.c = float(self.c)
        if not np.isfinite(self.c) or self.c < 0.0:
            raise ValueError("c must be finite and non-negative")

    @property
    def name(self) -> str:
        return f"UCB-SW(tau={self.window}, c={self.c:.5g})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        del rng
        self._sums = np.zeros(n_arms)
        self._rewards = [deque(maxlen=self.window) for _ in range(n_arms)]
        self._pulls = np.zeros(n_arms, dtype=int)

    def _window_pulls(self) -> np.ndarray:
        return np.fromiter((len(buffer) for buffer in self._rewards), dtype=int)

    def select_arm(self, step: int) -> int:
        counts = self._window_pulls()
        untried = np.flatnonzero(counts == 0)
        if untried.size:
            return int(untried[0])
        t = int(step) + 1
        scale = np.log(min(float(t), float(self.window)))
        scores = self._sums / counts + self.c * np.sqrt(scale / counts)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        buffer = self._rewards[arm]
        if len(buffer) == buffer.maxlen:
            self._sums[arm] -= buffer[0]
        buffer.append(reward)
        self._sums[arm] += reward
        self._pulls[arm] += 1

    @property
    def estimated_values(self) -> np.ndarray:
        counts = self._window_pulls()
        return self._sums / np.maximum(counts, 1)

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
