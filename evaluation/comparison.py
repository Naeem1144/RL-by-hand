"""Run algorithms alone or side by side on matched problem instances."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from algorithms.base import BanditAlgorithm
from evaluation.runner import RunResult, run
from problems.base import BanditProblem

ProblemSource = BanditProblem | Callable[[int], BanditProblem]

Z_95 = 1.96


@dataclass(frozen=True)
class AlgorithmSummary:
    """Aggregate outcome for one algorithm across repeated runs."""

    name: str
    mean_reward: float
    reward_std: float
    mean_regret: float
    regret_std: float


@dataclass(frozen=True)
class PairedComparison:
    """Per-seed paired outcome of one configuration against a baseline.

    ``mean_regret_difference`` is the baseline regret minus this
    configuration's regret, so positive values mean the configuration beats
    the baseline on the same seed.  Because both runs share a seed the
    comparison is paired, which is far more sensitive than comparing two
    independent confidence intervals.
    """

    baseline: str
    name: str
    n_seeds: int
    mean_regret_difference: float
    ci95_half_width: float
    win_rate: float


@dataclass(frozen=True)
class ComparisonResult:
    """All individual histories from a matched algorithm comparison."""

    runs: Mapping[str, tuple[RunResult, ...]]

    def summaries(self) -> tuple[AlgorithmSummary, ...]:
        rows: list[AlgorithmSummary] = []
        for name, results in self.runs.items():
            rewards = np.asarray([result.average_reward for result in results])
            regrets = np.asarray([result.total_regret for result in results])
            ddof = 1 if len(results) > 1 else 0
            rows.append(
                AlgorithmSummary(
                    name=name,
                    mean_reward=float(rewards.mean()),
                    reward_std=float(rewards.std(ddof=ddof)),
                    mean_regret=float(regrets.mean()),
                    regret_std=float(regrets.std(ddof=ddof)),
                )
            )
        return tuple(rows)

    @property
    def best_by_regret(self) -> AlgorithmSummary:
        return min(self.summaries(), key=lambda row: row.mean_regret)

    def paired_against(self, baseline: str) -> tuple[PairedComparison, ...]:
        """Compare every configuration with ``baseline`` seed by seed."""
        if baseline not in self.runs:
            raise KeyError(f"unknown baseline: {baseline}")
        baseline_regrets = np.asarray(
            [result.total_regret for result in self.runs[baseline]], dtype=float
        )

        rows: list[PairedComparison] = []
        for name, results in self.runs.items():
            if name == baseline:
                continue
            regrets = np.asarray([result.total_regret for result in results], dtype=float)
            differences = baseline_regrets - regrets
            n_seeds = differences.size
            ddof = 1 if n_seeds > 1 else 0
            half_width = Z_95 * differences.std(ddof=ddof) / np.sqrt(n_seeds)
            rows.append(
                PairedComparison(
                    baseline=baseline,
                    name=name,
                    n_seeds=n_seeds,
                    mean_regret_difference=float(differences.mean()),
                    ci95_half_width=float(half_width),
                    win_rate=float(np.mean(differences > 0.0)),
                )
            )
        return tuple(sorted(rows, key=lambda row: row.mean_regret_difference, reverse=True))


def compare(
    algorithms: Mapping[str, BanditAlgorithm],
    problem: ProblemSource,
    n_steps: int,
    *,
    seeds: Sequence[int] = (0,),
) -> ComparisonResult:
    """Evaluate named algorithms with matched problem and random seeds.

    ``problem`` may be one fixed problem or a factory receiving each seed.
    A factory is useful when performance should be averaged over multiple
    generated problem instances, and it is required for mutating problems:
    the factory is called once per seed and algorithm so no run ever shares
    problem state with another.  Matched seeds make every algorithm see the
    same instance and reward stream for a given seed.
    """
    if not algorithms:
        raise ValueError("at least one algorithm is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(not name for name in algorithms):
        raise ValueError("algorithm names must not be empty")

    collected: dict[str, list[RunResult]] = {name: [] for name in algorithms}
    for seed in seeds:
        for name, algorithm in algorithms.items():
            current_problem = problem(seed) if callable(problem) else problem
            collected[name].append(run(algorithm, current_problem, n_steps, seed=seed))

    return ComparisonResult({name: tuple(results) for name, results in collected.items()})
