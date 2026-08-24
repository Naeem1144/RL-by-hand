"""Isolated tests for Thompson sampling."""

import numpy as np
import pytest

from algorithms import ThompsonSampling
from evaluation import run
from problems import BernoulliBandit


def test_estimates_are_posterior_means() -> None:
    algorithm = ThompsonSampling(prior_alpha=2.0, prior_beta=3.0)
    result = run(algorithm, BernoulliBandit.random(5, seed=3), n_steps=40, seed=9)
    successes = np.bincount(result.selected_arms, weights=result.rewards, minlength=5)
    expected = (successes + 2.0) / (result.total_pulls + 5.0)
    assert np.allclose(result.estimated_values, expected)


def test_algorithm_can_be_exercised_without_a_problem() -> None:
    algorithm = ThompsonSampling(prior_alpha=2.0, prior_beta=3.0)
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    algorithm.update(arm=0, reward=1.0)

    assert np.allclose(algorithm.estimated_values, np.array([3 / 6, 2 / 5]))
    assert np.array_equal(algorithm.total_pulls, np.array([1, 0]))


def test_invalid_prior_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        ThompsonSampling(prior_alpha=np.inf)


def test_non_binary_reward_is_rejected() -> None:
    algorithm = ThompsonSampling()
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    with pytest.raises(ValueError, match="binary"):
        algorithm.update(arm=0, reward=0.5)
