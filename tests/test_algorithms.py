"""Correctness and validation tests for the public bandit runners."""

from collections.abc import Callable

import numpy as np
import pytest

from algorithms.epsilon_greedy import run_bandit
from algorithms.thompson import run_thompson
from algorithms.ucb import run_ucb
from bandit_utils import summarize

Runner = Callable[..., dict[str, np.ndarray]]


@pytest.mark.parametrize("runner", [run_bandit, run_ucb, run_thompson])
def test_result_invariants_and_determinism(runner: Runner) -> None:
    first = runner(n_arms=7, n_steps=50, seed=123)
    second = runner(n_arms=7, n_steps=50, seed=123)

    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert first["rewards"].shape == (50,)
    assert first["selected_arms"].shape == (50,)
    assert first["true_probs"].shape == (7,)
    assert first["estimated_values"].shape == (7,)
    assert first["total_pulls"].shape == (7,)
    assert first["total_pulls"].sum() == 50
    assert np.array_equal(
        first["total_pulls"],
        np.bincount(first["selected_arms"], minlength=7),
    )

    _, regret, _, _ = summarize(first)
    assert np.all(np.diff(regret) >= -1e-15)
    assert regret[-1] >= 0.0


@pytest.mark.parametrize("runner", [run_bandit, run_ucb, run_thompson])
def test_explicit_problem_is_preserved_and_not_mutated(runner: Runner) -> None:
    probabilities = np.array([0.1, 0.4, 0.9])
    original = probabilities.copy()
    result = runner(
        n_arms=3,
        n_steps=20,
        true_probs=probabilities,
        policy_seed=11,
        reward_seed=12,
    )

    assert np.array_equal(probabilities, original)
    assert np.array_equal(result["true_probs"], original)


@pytest.mark.parametrize("runner", [run_bandit, run_ucb])
def test_frequentist_estimates_equal_empirical_means(runner: Runner) -> None:
    result = runner(n_arms=5, n_steps=40, seed=9)
    reward_sums = np.bincount(result["selected_arms"], weights=result["rewards"], minlength=5)
    expected = np.divide(
        reward_sums,
        result["total_pulls"],
        out=np.zeros(5),
        where=result["total_pulls"] > 0,
    )
    assert np.allclose(result["estimated_values"], expected)


def test_thompson_estimates_are_posterior_means() -> None:
    result = run_thompson(
        n_arms=5,
        n_steps=40,
        seed=9,
        prior_alpha=2.0,
        prior_beta=3.0,
    )
    successes = np.bincount(result["selected_arms"], weights=result["rewards"], minlength=5)
    expected = (successes + 2.0) / (result["total_pulls"] + 5.0)
    assert np.allclose(result["estimated_values"], expected)


def test_ucb_bootstrap_pulls_each_arm_once() -> None:
    result = run_ucb(n_arms=5, n_steps=5, seed=3)
    assert np.array_equal(result["selected_arms"], np.arange(5))
    assert np.array_equal(result["total_pulls"], np.ones(5, dtype=int))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: run_bandit(2, 2, epsilon=1.1), "epsilon"),
        (lambda: run_bandit(2, 2, decay=True), "decay_rate"),
        (lambda: run_bandit(2, 2, decay_rate=0.9), "decay_rate"),
        (lambda: run_bandit(2, 2, decay="yes"), "boolean"),
        (lambda: run_bandit(2, 2, decay=True, decay_rate=1.0), "strictly"),
        (lambda: run_ucb(2, 3, c=-0.1), "non-negative"),
        (lambda: run_thompson(2, 3, prior_alpha=np.inf), "positive"),
        (lambda: run_thompson(0, 3), "n_arms"),
        (lambda: run_ucb(2, 0), "n_steps"),
        (
            lambda: run_bandit(2, 3, true_probs=np.array([0.2, 1.2])),
            "true_probs",
        ),
    ],
)
def test_invalid_inputs_fail_early(call: Callable[[], object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()
