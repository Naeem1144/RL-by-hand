"""Tests for the drifting Bernoulli bandit problem."""

import numpy as np
import pytest

from problems import DriftingBernoulliBandit


def test_probabilities_drift_after_every_sample() -> None:
    problem = DriftingBernoulliBandit(np.array([0.5, 0.5]), drift_std=0.1)
    before = problem.expected_rewards.copy()
    rng = np.random.default_rng(0)

    problem.sample(0, rng)

    assert not np.array_equal(problem.expected_rewards, before)


def test_expected_rewards_stay_within_bounds() -> None:
    problem = DriftingBernoulliBandit(np.full(4, 0.5), drift_std=0.2)
    rng = np.random.default_rng(1)
    for step in range(500):
        problem.sample(step % 4, rng)
        current = problem.expected_rewards
        assert np.all(current >= 0.02)
        assert np.all(current <= 0.98)


def test_matched_rng_reproduces_the_same_trajectory() -> None:
    def trajectory(problem: DriftingBernoulliBandit, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        for _ in range(50):
            problem.sample(0, rng)
        return problem.expected_rewards

    first = trajectory(DriftingBernoulliBandit(np.full(3, 0.5), drift_std=0.05), 7)
    second = trajectory(DriftingBernoulliBandit(np.full(3, 0.5), drift_std=0.05), 7)

    assert np.allclose(first, second)


def test_initial_probabilities_are_not_mutated() -> None:
    probabilities = np.array([0.3, 0.6])
    original = probabilities.copy()
    problem = DriftingBernoulliBandit(probabilities, drift_std=0.1)

    problem.sample(0, np.random.default_rng(2))

    assert np.array_equal(probabilities, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_probabilities": np.array([])},
        {"initial_probabilities": np.array([0.5, 1.5])},
        {"initial_probabilities": np.array([np.nan])},
        {"initial_probabilities": np.array([0.5]), "drift_std": 0.0},
        {"initial_probabilities": np.array([0.5]), "drift_std": -0.1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DriftingBernoulliBandit(**kwargs)
