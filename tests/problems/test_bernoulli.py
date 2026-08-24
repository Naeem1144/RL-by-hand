"""Tests for the Bernoulli bandit problem."""

import numpy as np
import pytest

from problems import BernoulliBandit


def test_sample_returns_only_zero_or_one_with_the_right_mean() -> None:
    problem = BernoulliBandit(np.array([0.0, 0.6, 1.0]))
    rng = np.random.default_rng(5)
    draws = np.array([problem.sample(1, rng) for _ in range(4_000)])

    assert set(draws) <= {0.0, 1.0}
    assert abs(draws.mean() - 0.6) < 0.05


def test_expected_rewards_match_probabilities_and_are_read_only() -> None:
    probabilities = np.array([0.2, 0.8])
    problem = BernoulliBandit(probabilities)

    assert np.array_equal(problem.expected_rewards, probabilities)
    with pytest.raises(ValueError):
        problem.expected_rewards[0] = 9.0


def test_random_factory_is_reproducible_and_bounded() -> None:
    first = BernoulliBandit.random(6, seed=11)
    second = BernoulliBandit.random(6, seed=11)

    assert first.n_arms == 6
    assert np.array_equal(first.probabilities, second.probabilities)
    assert np.all((first.probabilities >= 0.0) & (first.probabilities <= 1.0))


def test_sampling_does_not_mutate_the_input() -> None:
    probabilities = np.array([0.3, 0.7])
    original = probabilities.copy()
    problem = BernoulliBandit(probabilities)
    rng = np.random.default_rng(1)

    for _ in range(50):
        problem.sample(0, rng)

    assert np.array_equal(probabilities, original)
    assert np.array_equal(problem.probabilities, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probabilities": np.array([])},
        {"probabilities": np.array([[0.5, 0.5]])},
        {"probabilities": np.array([np.nan, 0.5])},
        {"probabilities": np.array([-0.1, 0.5])},
        {"probabilities": np.array([0.5, 1.1])},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BernoulliBandit(**kwargs)


@pytest.mark.parametrize("n_arms", [0, -1, True, 2.5])
def test_random_factory_rejects_invalid_arm_counts(n_arms) -> None:
    with pytest.raises((TypeError, ValueError)):
        BernoulliBandit.random(n_arms)


def test_out_of_range_or_non_integer_arm_is_rejected() -> None:
    problem = BernoulliBandit(np.array([0.1, 0.9]))
    with pytest.raises(IndexError):
        problem.sample(2, np.random.default_rng(0))
    with pytest.raises(TypeError):
        problem.sample(0.5, np.random.default_rng(0))
