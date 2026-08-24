"""Interchangeable multi-armed bandit algorithms."""

from .base import BanditAlgorithm
from .epsilon_greedy import EpsilonGreedy
from .forgetting_thompson import ForgettingThompsonSampling
from .klucb import KLUCB
from .sliding_window_ucb import SlidingWindowUCB
from .thompson import ThompsonSampling
from .thompson_normal import NormalThompsonSampling
from .ucb import UCB
from .ucb_variance import UCBVariance

__all__ = [
    "BanditAlgorithm",
    "EpsilonGreedy",
    "ForgettingThompsonSampling",
    "KLUCB",
    "NormalThompsonSampling",
    "SlidingWindowUCB",
    "ThompsonSampling",
    "UCB",
    "UCBVariance",
]
