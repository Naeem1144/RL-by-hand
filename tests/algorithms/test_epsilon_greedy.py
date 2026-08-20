"""Isolated tests for epsilon-greedy."""

import numpy as np
import pytest

from algorithms import EpsilonGreedy
from evaluation import run
from problems import BernoulliBandit


def test_estimates_are_empirical_means() -> None:
    problem = BernoulliBandit.random(n_arms=5, seed=3)
    result = run(EpsilonGreedy(), problem, n_steps=40, seed=9)
    reward_sums = np.bincount(result.selected_arms, weights=result.rewards, minlength=5)
    expected = np.divide(
        reward_sums,
        result.total_pulls,
        out=np.zeros(5),
        where=result.total_pulls > 0,
    )
    assert np.allclose(result.estimated_values, expected)


def test_algorithm_can_be_exercised_without_a_problem() -> None:
    algorithm = EpsilonGreedy(epsilon=0.0)
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    algorithm.update(arm=1, reward=1.0)

    assert algorithm.select_arm(step=1) == 1
    assert np.array_equal(algorithm.total_pulls, np.array([0, 1]))


def test_step_size_gives_recency_weighted_estimates() -> None:
    algorithm = EpsilonGreedy(epsilon=0.0, step_size=0.5)
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    algorithm.update(arm=0, reward=1.0)
    algorithm.update(arm=0, reward=0.0)

    assert np.allclose(algorithm.estimated_values, np.array([0.25, 0.0]))


def test_optimistic_value_initializes_all_arms_high() -> None:
    algorithm = EpsilonGreedy(epsilon=0.0, optimistic_value=2.5)
    algorithm.reset(n_arms=3, rng=np.random.default_rng(4))

    assert np.allclose(algorithm.estimated_values, np.full(3, 2.5))


@pytest.mark.parametrize(
    "configuration",
    [
        {"epsilon": 1.1},
        {"decay_rate": 1.0},
        {"step_size": 0.0},
        {"step_size": 1.5},
        {"optimistic_value": "high"},
        {"optimistic_value": np.nan},
    ],
)
def test_invalid_configuration_fails_early(configuration: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        EpsilonGreedy(**configuration)
