"""Isolated tests for upper confidence bound."""

import numpy as np
import pytest

from algorithms import UCB
from evaluation import run
from problems import BernoulliBandit


def test_bootstrap_pulls_each_arm_once() -> None:
    result = run(UCB(), BernoulliBandit.random(5, seed=2), n_steps=5, seed=3)
    assert np.array_equal(result.selected_arms, np.arange(5))
    assert np.array_equal(result.total_pulls, np.ones(5, dtype=int))


def test_estimates_are_empirical_means() -> None:
    result = run(UCB(), BernoulliBandit.random(5, seed=3), n_steps=40, seed=9)
    reward_sums = np.bincount(result.selected_arms, weights=result.rewards, minlength=5)
    expected = reward_sums / result.total_pulls
    assert np.allclose(result.estimated_values, expected)


def test_algorithm_can_be_exercised_without_a_problem() -> None:
    algorithm = UCB()
    algorithm.reset(n_arms=3, rng=np.random.default_rng(4))
    algorithm.update(0, 1.0)
    assert algorithm.select_arm(step=1) == 1


def test_negative_confidence_scale_is_invalid() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        UCB(c=-0.1)
