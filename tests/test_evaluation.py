"""Tests for the reusable single-run and head-to-head evaluation APIs."""

import numpy as np
import pytest

from algorithms import UCB, EpsilonGreedy, ThompsonSampling
from evaluation import compare, run
from problems import BernoulliBandit, DriftingBernoulliBandit


def test_run_solves_an_explicit_problem() -> None:
    problem = BernoulliBandit(np.array([0.0, 1.0]), name="certain winner")
    result = run(UCB(c=0.1), problem, n_steps=20, seed=7)

    assert result.problem_name == "certain winner"
    assert result.average_reward > 0.5
    assert result.rewards.shape == (20,)
    assert result.total_pulls.sum() == 20


def test_regret_uses_per_step_expected_rewards() -> None:
    problem = BernoulliBandit(np.array([0.2, 0.8]))
    result = run(EpsilonGreedy(epsilon=0.0), problem, n_steps=10, seed=1)

    assert np.allclose(result.expected_reward_of_best, 0.8)
    assert np.array_equal(
        result.expected_reward_of_selected, problem.expected_rewards[result.selected_arms]
    )
    assert np.allclose(
        result.cumulative_regret,
        np.cumsum(result.expected_reward_of_best - result.expected_reward_of_selected),
    )


def test_algorithms_can_be_compared_on_the_same_problem() -> None:
    problem = BernoulliBandit(np.array([0.1, 0.3, 0.8]))
    result = compare(
        {
            "epsilon": EpsilonGreedy(epsilon=0.1),
            "ucb": UCB(c=0.1),
            "thompson": ThompsonSampling(),
        },
        problem,
        n_steps=100,
        seeds=range(3),
    )

    assert set(result.runs) == {"epsilon", "ucb", "thompson"}
    assert all(len(runs) == 3 for runs in result.runs.values())
    assert {row.name for row in result.summaries()} == set(result.runs)
    assert result.best_by_regret.name in result.runs


def test_problem_factory_runs_once_per_algorithm_and_seed_and_is_matched() -> None:
    calls: list[int] = []

    def problem_factory(seed: int) -> DriftingBernoulliBandit:
        calls.append(seed)
        return DriftingBernoulliBandit(np.full(4, 0.5), drift_std=0.05)

    result = compare(
        {"one": EpsilonGreedy(), "two": UCB()},
        problem_factory,
        n_steps=30,
        seeds=[10, 11],
    )

    assert calls == [10, 10, 11, 11]
    assert np.array_equal(
        result.runs["one"][0].expected_reward_of_best,
        result.runs["two"][0].expected_reward_of_best,
    )
    assert not np.array_equal(
        result.runs["one"][0].expected_reward_of_best,
        result.runs["one"][1].expected_reward_of_best,
    )


def test_paired_comparison_aligns_runs_by_seed() -> None:
    problem = BernoulliBandit(np.array([0.0, 1.0]))
    result = compare(
        {"explorer": EpsilonGreedy(epsilon=1.0), "ucb": UCB(c=0.5)},
        problem,
        n_steps=200,
        seeds=range(4),
    )

    rows = result.paired_against("explorer")
    assert [row.name for row in rows] == ["ucb"]
    row = rows[0]
    assert row.baseline == "explorer"
    assert row.n_seeds == 4
    assert row.mean_regret_difference > 0.0
    assert row.win_rate == 1.0

    with pytest.raises(KeyError):
        result.paired_against("missing")
