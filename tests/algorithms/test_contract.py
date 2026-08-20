"""Contract tests that every algorithm implementation must pass."""

from collections.abc import Callable

import numpy as np
import pytest

from algorithms import UCB, EpsilonGreedy, NormalThompsonSampling, ThompsonSampling
from algorithms.base import BanditAlgorithm
from evaluation import run
from problems import BernoulliBandit

AlgorithmFactory = Callable[[], BanditAlgorithm]


@pytest.mark.parametrize(
    "factory",
    [EpsilonGreedy, UCB, ThompsonSampling, NormalThompsonSampling],
)
def test_algorithm_satisfies_shared_run_contract(factory: AlgorithmFactory) -> None:
    problem = BernoulliBandit(np.array([0.1, 0.4, 0.9]))
    first = run(factory(), problem, n_steps=50, seed=123)
    second = run(factory(), problem, n_steps=50, seed=123)

    assert np.array_equal(first.rewards, second.rewards)
    assert np.array_equal(first.selected_arms, second.selected_arms)
    assert first.rewards.shape == (50,)
    assert first.selected_arms.shape == (50,)
    assert first.estimated_values.shape == (3,)
    assert first.total_pulls.sum() == 50
    assert np.array_equal(first.total_pulls, np.bincount(first.selected_arms, minlength=3))
    assert np.all(np.diff(first.cumulative_regret) >= -1e-15)
    assert first.total_regret >= 0.0


@pytest.mark.parametrize(
    "factory",
    [EpsilonGreedy, UCB, ThompsonSampling, NormalThompsonSampling],
)
def test_problem_is_not_mutated(factory: AlgorithmFactory) -> None:
    probabilities = np.array([0.1, 0.4, 0.9])
    original = probabilities.copy()
    problem = BernoulliBandit(probabilities)

    run(factory(), problem, n_steps=20, seed=11)

    assert np.array_equal(probabilities, original)
    assert np.array_equal(problem.probabilities, original)
