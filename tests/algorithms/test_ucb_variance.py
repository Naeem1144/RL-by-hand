"""Isolated tests for variance-aware UCB."""

import numpy as np
import pytest

from algorithms import UCBVariance
from evaluation import run
from problems import BernoulliBandit, GaussianBandit


def test_estimates_are_empirical_means() -> None:
    result = run(UCBVariance(), BernoulliBandit.random(5, seed=3), n_steps=40, seed=9)
    reward_sums = np.bincount(result.selected_arms, weights=result.rewards, minlength=5)
    expected = np.divide(
        reward_sums,
        result.total_pulls,
        out=np.zeros(5),
        where=result.total_pulls > 0,
    )
    assert np.allclose(result.estimated_values, expected)


def test_variance_follows_welford() -> None:
    algorithm = UCBVariance()
    algorithm.reset(n_arms=1, rng=np.random.default_rng(4))
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0]
    for reward in rewards:
        algorithm.update(0, reward)

    assert np.allclose(algorithm._m2 / algorithm.total_pulls, np.var(rewards))


def test_algorithm_can_be_exercised_without_a_problem() -> None:
    algorithm = UCBVariance()
    algorithm.reset(n_arms=3, rng=np.random.default_rng(4))
    algorithm.update(0, 1.0)

    assert algorithm.select_arm(step=1) == 1


def test_solves_an_easy_gaussian_problem() -> None:
    problem = GaussianBandit(np.array([-2.0, 2.0]), reward_std=1.0)
    result = run(UCBVariance(), problem, n_steps=200, seed=5)

    assert result.average_reward > 1.0
    assert result.total_regret < 100.0


@pytest.mark.parametrize("configuration", [{"c": -0.1}, {"b": np.nan}, {"c": np.inf}])
def test_invalid_configuration_is_rejected(configuration: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UCBVariance(**configuration)
