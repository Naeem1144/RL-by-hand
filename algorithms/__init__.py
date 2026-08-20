"""Interchangeable multi-armed bandit algorithms."""

from .base import BanditAlgorithm
from .epsilon_greedy import EpsilonGreedy
from .thompson import ThompsonSampling
from .thompson_normal import NormalThompsonSampling
from .ucb import UCB

__all__ = [
    "BanditAlgorithm",
    "EpsilonGreedy",
    "NormalThompsonSampling",
    "ThompsonSampling",
    "UCB",
]
