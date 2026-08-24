"""Isolated tests for sliding-window UCB."""

import numpy as np
import pytest

from algorithms import SlidingWindowUCB
from evaluation import run
from problems import BernoulliBandit


def test_estimates_use_only_the_window() -> None:
    algorithm = SlidingWindowUCB(window=3)
    algorithm.reset(n_arms=1, rng=np.random.default_rng(4))
    for reward in [1.0, 0.0, 0.0, 0.0, 0.0]:
        algorithm.update(0, reward)

    assert np.allclose(algorithm.estimated_values, np.array([0.0]))


def test_window_mean_when_partially_full() -> None:
    algorithm = SlidingWindowUCB(window=5)
    algorithm.reset(n_arms=1, rng=np.random.default_rng(4))
    for reward in [1.0, 1.0, 1.0]:
        algorithm.update(0, reward)

    assert np.allclose(algorithm.estimated_values, np.array([1.0]))
    for reward in [1.0, 0.0]:
        algorithm.update(0, reward)
    assert np.allclose(algorithm.estimated_values, np.array([0.8]))


def test_total_pulls_are_all_time() -> None:
    algorithm = SlidingWindowUCB(window=2)
    algorithm.reset(n_arms=1, rng=np.random.default_rng(4))
    for _ in range(7):
        algorithm.update(0, 1.0)

    assert algorithm.total_pulls[0] == 7
    assert algorithm.estimated_values[0] == 1.0


def test_bootstraps_each_arm_once() -> None:
    result = run(SlidingWindowUCB(window=5), BernoulliBandit.random(5, seed=2), n_steps=5, seed=3)
    assert np.array_equal(result.selected_arms, np.arange(5))


@pytest.mark.parametrize("window", [0, -3])
def test_invalid_window_is_rejected(window: int) -> None:
    with pytest.raises(ValueError):
        SlidingWindowUCB(window=window)


def test_non_integer_window_is_rejected() -> None:
    with pytest.raises(TypeError):
        SlidingWindowUCB(window=2.5)


def test_invalid_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        SlidingWindowUCB(c=-1.0)
