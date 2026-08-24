"""Tests for the Gaussian bandit problem."""

import numpy as np
import pytest

from problems import GaussianBandit


def test_sample_draws_around_the_arm_mean() -> None:
    problem = GaussianBandit(np.array([-1.0, 2.0]), reward_std=0.5)
    rng = np.random.default_rng(3)
    draws = np.array([problem.sample(1, rng) for _ in range(4_000)])

    assert abs(draws.mean() - 2.0) < 0.05
    assert abs(draws.std() - 0.5) < 0.05


def test_expected_rewards_match_means_and_are_read_only() -> None:
    means = np.array([0.5, -0.5])
    problem = GaussianBandit(means)

    assert np.array_equal(problem.expected_rewards, means)
    with pytest.raises(ValueError):
        problem.expected_rewards[0] = 9.0


def test_random_factory_is_reproducible() -> None:
    first = GaussianBandit.random(6, seed=11)
    second = GaussianBandit.random(6, seed=11)

    assert first.n_arms == 6
    assert np.array_equal(first.means, second.means)
    assert np.all((first.means >= -1.0) & (first.means <= 3.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"means": np.array([])},
        {"means": np.array([[1.0, 2.0]])},
        {"means": np.array([np.nan, 1.0])},
        {"means": np.array([0.0, 1.0]), "reward_std": 0.0},
        {"means": np.array([0.0, 1.0]), "reward_std": -1.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GaussianBandit(**kwargs)


def test_out_of_range_arm_is_rejected() -> None:
    problem = GaussianBandit(np.array([0.0, 1.0]))
    with pytest.raises(IndexError):
        problem.sample(2, np.random.default_rng(0))
