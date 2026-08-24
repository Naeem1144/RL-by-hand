"""Isolated tests for forgetting Thompson sampling."""

import numpy as np
import pytest

from algorithms import ForgettingThompsonSampling, ThompsonSampling


def test_counts_decay_geometrically() -> None:
    algorithm = ForgettingThompsonSampling(gamma=0.9, prior_alpha=1.0, prior_beta=1.0)
    algorithm.reset(n_arms=1, rng=np.random.default_rng(4))
    algorithm.update(0, 1.0)
    for reward in [0.0] * 10:
        algorithm.update(0, reward)

    expected_successes = 0.9**10
    expected_failures = (1.0 - 0.9**10) / (1.0 - 0.9)
    expected = (expected_successes + 1.0) / (expected_successes + expected_failures + 2.0)
    assert np.allclose(algorithm.estimated_values, np.array([expected]))


def test_first_reward_decay_is_noop() -> None:
    algorithm = ForgettingThompsonSampling(gamma=0.5)
    algorithm.reset(n_arms=1, rng=np.random.default_rng(4))
    algorithm.update(0, 1.0)

    assert np.allclose(algorithm.estimated_values, np.array([2.0 / 3.0]))


def test_gamma_one_matches_stationary_thompson() -> None:
    forgetting = ForgettingThompsonSampling(gamma=1.0, prior_alpha=2.0, prior_beta=3.0)
    stationary = ThompsonSampling(prior_alpha=2.0, prior_beta=3.0)
    for algorithm in (forgetting, stationary):
        algorithm.reset(n_arms=1, rng=np.random.default_rng(7))
        for reward in [1.0, 0.0, 1.0, 1.0]:
            algorithm.update(0, reward)

    assert np.allclose(forgetting.estimated_values, stationary.estimated_values)


def test_non_binary_reward_is_rejected() -> None:
    algorithm = ForgettingThompsonSampling()
    algorithm.reset(n_arms=2, rng=np.random.default_rng(4))
    with pytest.raises(ValueError, match="binary"):
        algorithm.update(arm=0, reward=0.5)


@pytest.mark.parametrize("gamma", [0.0, -0.1, 1.5, np.nan])
def test_invalid_gamma_is_rejected(gamma: float) -> None:
    with pytest.raises(ValueError):
        ForgettingThompsonSampling(gamma=gamma)
