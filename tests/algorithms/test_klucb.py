"""Isolated tests for KL-UCB."""

import numpy as np
import pytest

from algorithms import KLUCB
from algorithms.klucb import _kl_ucb_index
from evaluation import run
from problems import BernoulliBandit


def test_estimates_are_empirical_means() -> None:
    result = run(KLUCB(), BernoulliBandit.random(5, seed=3), n_steps=40, seed=9)
    reward_sums = np.bincount(result.selected_arms, weights=result.rewards, minlength=5)
    expected = reward_sums / result.total_pulls
    assert np.allclose(result.estimated_values, expected)


def test_algorithm_can_be_exercised_without_a_problem() -> None:
    algorithm = KLUCB()
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    algorithm.update(arm=0, reward=1.0)

    assert algorithm.select_arm(step=1) == 1  # arm 1 is still untried
    algorithm.update(arm=1, reward=0.0)
    assert algorithm.select_arm(step=2) in (0, 1)


def test_index_reaches_one_for_always_successful_arms() -> None:
    index = _kl_ucb_index(np.array([1.0]), np.array([10.0]), budget=1.0)
    assert np.allclose(index, np.array([1.0]))


def test_index_matches_analytic_solution_for_zero_mean() -> None:
    mean = 0.0
    n = 10.0
    budget = 2.0
    # kl(0 || theta) = -log(1 - theta); solving yields theta = 1 - exp(-budget / n)
    expected = 1.0 - np.exp(-budget / n)
    index = _kl_ucb_index(np.array([mean]), np.array([n]), budget=budget)
    assert np.allclose(index, np.array([expected]), atol=1e-12)


def test_index_is_monotone_in_observed_mean() -> None:
    low = _kl_ucb_index(np.array([0.2]), np.array([100.0]), budget=5.0)[0]
    high = _kl_ucb_index(np.array([0.5]), np.array([100.0]), budget=5.0)[0]
    assert high > low
    assert low > 0.2 and high > 0.5


def test_solves_an_easy_problem() -> None:
    problem = BernoulliBandit(np.array([0.1, 0.9]))
    result = run(KLUCB(), problem, n_steps=2_000, seed=5)

    assert result.average_reward > 0.6
    assert result.total_regret < 200.0


def test_non_binary_reward_is_rejected() -> None:
    algorithm = KLUCB()
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    with pytest.raises(ValueError, match="binary"):
        algorithm.update(arm=0, reward=0.5)


@pytest.mark.parametrize("c", [-0.1, np.nan, np.inf])
def test_invalid_scale_is_rejected(c: float) -> None:
    with pytest.raises(ValueError):
        KLUCB(c=c)
