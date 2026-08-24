"""Interface implemented by every bandit algorithm."""

from abc import ABC, abstractmethod

import numpy as np


class BanditAlgorithm(ABC):
    """A policy that chooses arms and learns from observed rewards.

    Algorithms deliberately know nothing about the problem that generates a
    reward.  The shared evaluator owns that interaction loop, so adding a new
    policy only requires implementing this small interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in reports."""

    @abstractmethod
    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        """Reset all learned state for a new run."""

    @abstractmethod
    def select_arm(self, step: int) -> int:
        """Choose an arm for the zero-indexed simulation step."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Learn from the reward observed after selecting ``arm``."""

    @property
    @abstractmethod
    def estimated_values(self) -> np.ndarray:
        """Current expected-reward estimate for every arm."""

    @property
    @abstractmethod
    def total_pulls(self) -> np.ndarray:
        """Number of times each arm has been selected."""
