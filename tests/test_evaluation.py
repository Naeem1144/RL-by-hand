"""Tests for the reusable single-run and head-to-head evaluation APIs."""

import numpy as np
import pytest

from algorithms import UCB, EpsilonGreedy, ThompsonSampling
from algorithms.base import BanditAlgorithm
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


def test_paired_comparison_with_single_seed() -> None:
    problem = BernoulliBandit(np.array([0.0, 1.0]))
    result = compare(
        {"explorer": EpsilonGreedy(epsilon=1.0), "ucb": UCB(c=0.5)},
        problem,
        n_steps=50,
        seeds=[7],
    )

    rows = result.paired_against("explorer")
    assert [row.name for row in rows] == ["ucb"]
    assert rows[0].n_seeds == 1
    assert rows[0].ci95_half_width == 0.0  # ddof falls back to 0, no spread
    assert rows[0].win_rate in (0.0, 1.0)


def test_compare_rejects_shared_mutating_problem() -> None:
    problem = DriftingBernoulliBandit(np.full(4, 0.5), drift_std=0.05)
    with pytest.raises(ValueError, match="factory"):
        compare({"egreedy": EpsilonGreedy()}, problem, n_steps=10, seeds=[0])


class _FixedArm(BanditAlgorithm):
    """Selects one fixed arm; used to trigger runner validation paths."""

    def __init__(self, arm: int, record_pulls: bool = True) -> None:
        self._requested_arm = arm
        self._record_pulls = record_pulls
        self._pulls: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"fixed-arm({self._requested_arm})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        del rng
        self._pulls = np.zeros(n_arms, dtype=int)

    def select_arm(self, step: int) -> int:
        del step
        return self._requested_arm

    def update(self, arm: int, reward: float) -> None:
        if self._record_pulls:
            self._pulls[arm] += 1

    @property
    def estimated_values(self) -> np.ndarray:
        return np.zeros_like(self._pulls, dtype=float)

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls


class _NaNProblem:
    """Returns a non-finite reward to exercise runner validation."""

    name = "nan problem"

    @property
    def n_arms(self) -> int:
        return 1

    @property
    def expected_rewards(self) -> np.ndarray:
        return np.array([0.0])

    def sample(self, arm: int, rng: np.random.Generator) -> float:
        del arm, rng
        return float("nan")


def test_run_rejects_out_of_range_arm() -> None:
    problem = BernoulliBandit(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="expected an arm"):
        run(_FixedArm(2), problem, n_steps=5, seed=0)


def test_run_rejects_non_finite_rewards() -> None:
    with pytest.raises(ValueError, match="finite reward"):
        run(EpsilonGreedy(), _NaNProblem(), n_steps=5, seed=0)


def test_run_rejects_inconsistent_pull_counts() -> None:
    problem = BernoulliBandit(np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="total_pulls"):
        run(_FixedArm(0, record_pulls=False), problem, n_steps=5, seed=0)
