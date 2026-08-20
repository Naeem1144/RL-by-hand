"""Isolated tests for normal Thompson sampling."""

import numpy as np
import pytest

from algorithms import NormalThompsonSampling
from evaluation import run
from problems import GaussianBandit


def test_estimates_are_posterior_means() -> None:
    algorithm = NormalThompsonSampling(reward_std=1.0, prior_mean=0.0, prior_std=1.0)
    result = run(algorithm, GaussianBandit.random(5, seed=3), n_steps=40, seed=9)
    sums = np.bincount(result.selected_arms, weights=result.rewards, minlength=5)
    expected = sums / (result.total_pulls + 1.0)
    assert np.allclose(result.estimated_values, expected)


def test_algorithm_can_be_exercised_without_a_problem() -> None:
    algorithm = NormalThompsonSampling(prior_mean=0.0, prior_std=1.0, reward_std=2.0)
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    algorithm.update(arm=0, reward=3.0)

    precision = 1.0 + 1.0 / 4.0
    expected_mean = 3.0 / 4.0 / precision
    assert np.allclose(algorithm.estimated_values, np.array([expected_mean, 0.0]))
    assert np.array_equal(algorithm.total_pulls, np.array([1, 0]))


def test_solves_an_easy_gaussian_problem() -> None:
    problem = GaussianBandit(np.array([-2.0, 2.0]), reward_std=1.0)
    result = run(NormalThompsonSampling(), problem, n_steps=200, seed=5)

    assert result.average_reward > 1.0
    assert result.total_regret < 100.0


@pytest.mark.parametrize(
    "configuration",
    [
        {"reward_std": 0.0},
        {"reward_std": np.inf},
        {"prior_std": -1.0},
        {"prior_mean": np.nan},
    ],
)
def test_invalid_configuration_is_rejected(configuration: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        NormalThompsonSampling(**configuration)
