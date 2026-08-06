"""Tests for metrics, matched instances, and benchmark inference."""

import numpy as np
import pytest

import compare_hyperparameters as hyper
import compare_problem_suites as suites
from algorithms.epsilon_greedy import run_bandit
from bandit_utils import (
    BENCHMARK_ENVIRONMENT_SEEDS,
    average_over_seeds,
    generate_problem,
    simulation_seeds,
    summarize,
)


def test_problem_and_simulation_seeds_are_reproducible_and_separate() -> None:
    assert np.array_equal(generate_problem(5, 17), generate_problem(5, 17))
    policy_seed, reward_seed = simulation_seeds(17)
    assert policy_seed != reward_seed
    assert (policy_seed, reward_seed) == simulation_seeds(17)


def test_summarize_uses_pseudo_regret() -> None:
    results = {
        "rewards": np.array([0, 1, 0]),
        "selected_arms": np.array([0, 1, 0]),
        "true_probs": np.array([0.25, 0.75]),
    }
    steps, regret, reward, final_reward = summarize(results)
    assert np.array_equal(steps, np.array([1, 2, 3]))
    assert np.allclose(regret, np.array([0.5, 0.5, 1.0]))
    assert np.allclose(reward, np.array([0.0, 0.5, 1.0 / 3.0]))
    assert final_reward == pytest.approx(1.0 / 3.0)


def test_average_over_seeds_requires_variability_sample() -> None:
    with pytest.raises(ValueError, match="at least two"):
        average_over_seeds(
            lambda seed: run_bandit(n_arms=2, n_steps=2, seed=seed),
            [0],
        )


def test_average_over_seeds_returns_series_standard_errors() -> None:
    output = average_over_seeds(
        lambda seed: run_bandit(n_arms=3, n_steps=8, seed=seed),
        [0, 1, 2],
    )
    assert len(output) == 11
    assert output[2].shape == (8,)
    assert output[4].shape == (8,)
    assert np.all(output[2] >= 0.0)
    assert np.all(output[4] >= 0.0)


def test_simultaneous_inference_uses_fixed_reference() -> None:
    reference = np.array([10.0, 11.0, 9.0, 10.0])
    rows = [
        {
            "family": "family",
            "variant": "tuning-reference",
            "total_regret": 10.0,
            "se_regret": 0.4,
            "per_seed_regret": reference,
        },
        {
            "family": "family",
            "variant": "evaluation-winner",
            "total_regret": 9.0,
            "se_regret": 0.4,
            "per_seed_regret": reference - 1.0,
        },
        {
            "family": "family",
            "variant": "worse",
            "total_regret": 20.0,
            "se_regret": 0.4,
            "per_seed_regret": reference + np.array([9.0, 11.0, 10.0, 10.0]),
        },
    ]

    selected, critical = hyper.annotate_significance(rows, ("family", "tuning-reference"))
    assert selected["variant"] == "tuning-reference"
    assert critical >= 0.0
    assert rows[2]["sig_worse_than_reference"] is True


def test_hyperparameter_grid_has_stable_unique_identities() -> None:
    specs = hyper.build_run_specs()
    identities = [(family, variant) for family, variant, _, _ in specs]
    assert len(specs) == 36
    assert len(set(identities)) == len(identities)


@pytest.mark.parametrize("suite", suites.PROBLEM_SUITES)
def test_problem_suites_generate_valid_probabilities(suite: str) -> None:
    probabilities = suites.generate_suite_problem(suite, 31_337)
    assert probabilities.shape == (suites.N_ARMS,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.array_equal(probabilities, suites.generate_suite_problem(suite, 31_337))


def test_experiment_environment_splits_are_disjoint() -> None:
    groups = [
        set(hyper.TUNING_ENVIRONMENT_SEEDS),
        set(hyper.EVALUATION_ENVIRONMENT_SEEDS),
        set(BENCHMARK_ENVIRONMENT_SEEDS),
        set(suites.ENVIRONMENT_SEEDS),
    ]
    assert all(
        left.isdisjoint(right) for index, left in enumerate(groups) for right in groups[index + 1 :]
    )
