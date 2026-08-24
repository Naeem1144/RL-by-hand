"""Sweep each algorithm family's main hyperparameters on three benchmark scenarios."""

import csv
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms import (
    KLUCB,
    UCB,
    BanditAlgorithm,
    EpsilonGreedy,
    ForgettingThompsonSampling,
    NormalThompsonSampling,
    SlidingWindowUCB,
    ThompsonSampling,
    UCBVariance,
)
from evaluation import AlgorithmSummary, ComparisonResult, compare
from problems import BanditProblem, BernoulliBandit, DriftingBernoulliBandit, GaussianBandit

N_STEPS = 10_000
SEEDS = tuple(range(50))

PROBABILITIES = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55])
GAUSSIAN_MEANS = np.linspace(-1.0, 3.5, 10)
GAUSSIAN_REWARD_STD = 1.0
DRIFT_STD = 0.02

EPSILONS = (0.01, 0.05, 0.10, 0.20, 0.30)
DECAYED_EPSILONS = ((0.10, 0.999), (0.20, 0.995))
UCB_SCALES = (0.10, 0.50, 1.00, float(np.sqrt(2.0)), 2.00)
THOMPSON_PRIORS = (0.50, 1.00, 2.00, 5.00)

GAUSSIAN_EPSILONS = (0.05, 0.10, 0.20)
GAUSSIAN_UCB_SCALES = (0.50, 1.00, float(np.sqrt(2.0)))
NORMAL_PRIOR_STDS = (0.50, 1.00, 2.00)

DRIFTING_EPSILONS = (0.05, 0.10, 0.20)
STEP_SIZES = (0.05, 0.10, 0.20)
SLOW_DRIFT_STD = 0.002
SW_WINDOWS = (1_000, 4_000)
FORGETTING_GAMMAS = (0.95, 0.99)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
Z_95 = 1.96

FAMILY_COLORS = {
    "Epsilon-greedy": "#2563eb",
    "UCB": "#dc2626",
    "Thompson sampling": "#16a34a",
}


def bernoulli_configurations() -> dict[str, BanditAlgorithm]:
    """Configurations for the fixed Bernoulli scenario."""
    algorithms: dict[str, BanditAlgorithm] = {}
    for epsilon in EPSILONS:
        algorithms[f"Epsilon epsilon={epsilon:g}"] = EpsilonGreedy(epsilon=epsilon)
    for epsilon, decay in DECAYED_EPSILONS:
        algorithms[f"Epsilon epsilon={epsilon:g}, decay={decay:g}"] = EpsilonGreedy(
            epsilon=epsilon,
            decay_rate=decay,
        )
    algorithms["Epsilon epsilon=0, optimistic"] = EpsilonGreedy(
        epsilon=0.0,
        optimistic_value=1.0,
    )
    for c in UCB_SCALES:
        algorithms[f"UCB c={c:.3g}"] = UCB(c=c)
    for prior in THOMPSON_PRIORS:
        algorithms[f"Thompson Beta({prior:g},{prior:g})"] = ThompsonSampling(
            prior_alpha=prior,
            prior_beta=prior,
        )
    algorithms["KL-UCB c=3"] = KLUCB(c=3.0)
    algorithms["UCB-V(default)"] = UCBVariance()
    return algorithms


def gaussian_configurations() -> dict[str, BanditAlgorithm]:
    """Configurations for the fixed Gaussian scenario."""
    algorithms: dict[str, BanditAlgorithm] = {}
    for epsilon in GAUSSIAN_EPSILONS:
        algorithms[f"Epsilon epsilon={epsilon:g}"] = EpsilonGreedy(epsilon=epsilon)
    algorithms["Epsilon epsilon=0, optimistic"] = EpsilonGreedy(
        epsilon=0.0,
        optimistic_value=float(GAUSSIAN_MEANS.max()) + 0.5,
    )
    for c in GAUSSIAN_UCB_SCALES:
        algorithms[f"UCB c={c:.3g}"] = UCB(c=c)
    for prior_std in NORMAL_PRIOR_STDS:
        algorithms[f"Thompson-Normal sigma0={prior_std:g}"] = NormalThompsonSampling(
            reward_std=GAUSSIAN_REWARD_STD,
            prior_mean=0.0,
            prior_std=prior_std,
        )
    algorithms["UCB-V(default)"] = UCBVariance()
    return algorithms


def _base_drifting_configurations() -> dict[str, BanditAlgorithm]:
    """Configurations contrasting sample means with recency-weighted estimates."""
    algorithms: dict[str, BanditAlgorithm] = {}
    for epsilon in DRIFTING_EPSILONS:
        algorithms[f"Epsilon epsilon={epsilon:g}"] = EpsilonGreedy(epsilon=epsilon)
    for alpha in STEP_SIZES:
        algorithms[f"Epsilon epsilon=0.1, alpha={alpha:g}"] = EpsilonGreedy(
            epsilon=0.1,
            step_size=alpha,
        )
    algorithms["UCB c=1.41"] = UCB(c=float(np.sqrt(2.0)))
    algorithms["Thompson Beta(1,1)"] = ThompsonSampling(prior_alpha=1.0, prior_beta=1.0)
    return algorithms


def drifting_configurations() -> dict[str, BanditAlgorithm]:
    """Configurations for the fast-drifting scenario: stationary baselines plus
    drift-aware variants (sliding-window UCB, forgetting Thompson)."""
    algorithms = _base_drifting_configurations()
    algorithms["KL-UCB c=3"] = KLUCB(c=3.0)
    algorithms["UCB-V(default)"] = UCBVariance()
    algorithms["UCB SW(tau=1000)"] = SlidingWindowUCB(window=1_000)
    algorithms["Thompson decay=0.95"] = ForgettingThompsonSampling(gamma=0.95)
    algorithms["Thompson decay=0.99"] = ForgettingThompsonSampling(gamma=0.99)
    return algorithms


def slow_drifting_configurations() -> dict[str, BanditAlgorithm]:
    """Same families as the fast-drifting scenario, plus a longer window variant."""
    algorithms = _base_drifting_configurations()
    algorithms["KL-UCB c=3"] = KLUCB(c=3.0)
    algorithms["UCB-V(default)"] = UCBVariance()
    for window in SW_WINDOWS:
        algorithms[f"UCB SW(tau={window})"] = SlidingWindowUCB(window=window)
    for gamma in FORGETTING_GAMMAS:
        algorithms[f"Thompson decay={gamma:g}"] = ForgettingThompsonSampling(gamma=gamma)
    return algorithms


@dataclass(frozen=True)
class Scenario:
    """One benchmark problem plus the configurations evaluated on it."""

    key: str
    title: str
    make_problem: Callable[[int], BanditProblem]
    configurations: Callable[[], dict[str, BanditAlgorithm]]
    grids: dict[str, object]
    reference_reward: float | None


SCENARIOS = (
    Scenario(
        key="bernoulli",
        title="Fixed ten-arm Bernoulli",
        make_problem=lambda seed: BernoulliBandit(
            PROBABILITIES,
            name="fixed ten-arm Bernoulli bandit",
        ),
        configurations=bernoulli_configurations,
        grids={
            "epsilon": list(EPSILONS),
            "decayed_epsilon": [list(pair) for pair in DECAYED_EPSILONS],
            "optimistic_value": [1.0],
            "ucb_c": list(UCB_SCALES),
            "symmetric_beta_prior": list(THOMPSON_PRIORS),
            "kl_ucb_c": [3.0],
            "ucb_v": ["default"],
        },
        reference_reward=float(PROBABILITIES.max()),
    ),
    Scenario(
        key="gaussian",
        title="Fixed ten-arm Gaussian",
        make_problem=lambda seed: GaussianBandit(
            GAUSSIAN_MEANS,
            reward_std=GAUSSIAN_REWARD_STD,
            name="fixed ten-arm Gaussian bandit",
        ),
        configurations=gaussian_configurations,
        grids={
            "epsilon": list(GAUSSIAN_EPSILONS),
            "optimistic_value": [float(GAUSSIAN_MEANS.max()) + 0.5],
            "ucb_c": list(GAUSSIAN_UCB_SCALES),
            "normal_prior_std": list(NORMAL_PRIOR_STDS),
            "ucb_v": ["default"],
        },
        reference_reward=float(GAUSSIAN_MEANS.max()),
    ),
    Scenario(
        key="drifting",
        title="Drifting ten-arm Bernoulli",
        make_problem=lambda seed: DriftingBernoulliBandit(
            PROBABILITIES,
            drift_std=DRIFT_STD,
            name="drifting ten-arm Bernoulli bandit",
        ),
        configurations=drifting_configurations,
        grids={
            "epsilon": list(DRIFTING_EPSILONS),
            "step_size": list(STEP_SIZES),
            "ucb_c": [float(np.sqrt(2.0))],
            "symmetric_beta_prior": [1.0],
            "kl_ucb_c": [3.0],
            "ucb_v": ["default"],
            "sw_window": [1_000],
            "forgetting_gamma": list(FORGETTING_GAMMAS),
        },
        reference_reward=None,
    ),
    Scenario(
        key="slow_drifting",
        title="Slowly drifting ten-arm Bernoulli",
        make_problem=lambda seed: DriftingBernoulliBandit(
            PROBABILITIES,
            drift_std=SLOW_DRIFT_STD,
            name="slowly drifting ten-arm Bernoulli bandit",
        ),
        configurations=slow_drifting_configurations,
        grids={
            "epsilon": list(DRIFTING_EPSILONS),
            "step_size": list(STEP_SIZES),
            "ucb_c": [float(np.sqrt(2.0))],
            "symmetric_beta_prior": [1.0],
            "kl_ucb_c": [3.0],
            "ucb_v": ["default"],
            "sw_window": list(SW_WINDOWS),
            "forgetting_gamma": list(FORGETTING_GAMMAS),
        },
        reference_reward=None,
    ),
)


def family_of(configuration: str) -> str:
    """Return the algorithm family represented by a configuration label."""
    if configuration.startswith("Epsilon "):
        return "Epsilon-greedy"
    if configuration.startswith(("UCB ", "UCB-", "KL-UCB")):
        return "UCB"
    if configuration.startswith("Thompson"):
        return "Thompson sampling"
    raise ValueError(f"unknown configuration: {configuration}")


def run_study(
    n_steps: int = N_STEPS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, ComparisonResult]:
    """Run every scenario's matched hyperparameter sweep."""
    return {
        scenario.key: compare(
            scenario.configurations(),
            scenario.make_problem,
            n_steps=n_steps,
            seeds=seeds,
        )
        for scenario in SCENARIOS
    }


def _ci95(std: float, n_runs: int) -> float:
    return Z_95 * std / np.sqrt(n_runs)


def _seed_spec(seeds: Sequence[int]) -> dict[str, int] | list[int]:
    values = list(seeds)
    if values == list(range(values[0], values[-1] + 1)):
        return {"start": values[0], "stop": values[-1] + 1}
    return values


def _ranked(comparison: ComparisonResult) -> list[AlgorithmSummary]:
    return sorted(comparison.summaries(), key=lambda row: row.mean_regret)


def _best_by_family(comparison: ComparisonResult) -> dict[str, AlgorithmSummary]:
    best: dict[str, AlgorithmSummary] = {}
    for row in comparison.summaries():
        family = family_of(row.name)
        if family not in best or row.mean_regret < best[family].mean_regret:
            best[family] = row
    return best


def _write_summary(comparison: ComparisonResult, output_dir: Path) -> Path:
    n_runs = len(next(iter(comparison.runs.values())))
    path = output_dir / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "family",
                "configuration",
                "mean_reward",
                "reward_ci95_half_width",
                "mean_regret",
                "regret_ci95_half_width",
                "median_regret",
                "p90_regret",
                "max_regret",
            ]
        )
        for rank, row in enumerate(_ranked(comparison), start=1):
            writer.writerow(
                [
                    rank,
                    family_of(row.name),
                    row.name,
                    f"{row.mean_reward:.6f}",
                    f"{_ci95(row.reward_std, n_runs):.6f}",
                    f"{row.mean_regret:.6f}",
                    f"{_ci95(row.regret_std, n_runs):.6f}",
                    f"{row.median_regret:.6f}",
                    f"{row.p90_regret:.6f}",
                    f"{row.max_regret:.6f}",
                ]
            )
    return path


def _write_runs(
    comparison: ComparisonResult,
    seeds: Sequence[int],
    output_dir: Path,
) -> Path:
    path = output_dir / "per_run.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["family", "configuration", "seed", "average_reward", "total_regret"])
        for name, runs in comparison.runs.items():
            for seed, result in zip(seeds, runs, strict=True):
                writer.writerow(
                    [
                        family_of(name),
                        name,
                        seed,
                        f"{result.average_reward:.6f}",
                        f"{result.total_regret:.6f}",
                    ]
                )
    return path


def _write_paired(comparison: ComparisonResult, output_dir: Path) -> Path:
    baseline = comparison.best_by_regret.name
    path = output_dir / "paired.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "baseline",
                "configuration",
                "n_seeds",
                "mean_regret_difference",
                "ci95_half_width",
                "win_rate",
            ]
        )
        for row in comparison.paired_against(baseline):
            writer.writerow(
                [
                    baseline,
                    row.name,
                    row.n_seeds,
                    f"{row.mean_regret_difference:.6f}",
                    f"{row.ci95_half_width:.6f}",
                    f"{row.win_rate:.4f}",
                ]
            )
    return path


def _curve(runs, attribute: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([getattr(result, attribute) for result in runs])
    mean = values.mean(axis=0)
    if len(runs) == 1:
        return mean, np.zeros_like(mean)
    half_width = Z_95 * values.std(axis=0, ddof=1) / np.sqrt(len(runs))
    return mean, half_width


def _plot_endpoint_rankings(
    comparison: ComparisonResult,
    regret_axis,
    reward_axis,
) -> None:
    rows = list(reversed(_ranked(comparison)))
    n_runs = len(next(iter(comparison.runs.values())))
    positions = np.arange(len(rows))
    colors = [FAMILY_COLORS[family_of(row.name)] for row in rows]

    for position, row, color in zip(positions, rows, colors, strict=True):
        regret_axis.errorbar(
            row.mean_regret,
            position,
            xerr=_ci95(row.regret_std, n_runs),
            color=color,
            marker="o",
            capsize=3,
        )
        reward_axis.errorbar(
            row.mean_reward,
            position,
            xerr=_ci95(row.reward_std, n_runs),
            color=color,
            marker="o",
            capsize=3,
        )

    labels = [row.name for row in rows]
    regret_axis.set_yticks(positions, labels, fontsize=8)
    reward_axis.set_yticks(positions, [])
    regret_axis.set(
        title="Final cumulative pseudo-regret",
        xlabel="Mean regret with 95% CI (lower is better)",
    )
    reward_axis.set(
        title="Final average reward",
        xlabel="Mean reward with 95% CI (higher is better)",
    )


def _plot_best_family_curves(
    comparison: ComparisonResult,
    regret_axis,
    reward_axis,
    reference_reward: float | None,
) -> None:
    best = _best_by_family(comparison)
    for family in FAMILY_COLORS:
        row = best[family]
        runs = comparison.runs[row.name]
        color = FAMILY_COLORS[family]
        steps = runs[0].steps
        regret, regret_ci = _curve(runs, "cumulative_regret")
        reward, reward_ci = _curve(runs, "running_average_reward")
        label = row.name
        regret_axis.plot(steps, regret, color=color, label=label)
        regret_axis.fill_between(
            steps,
            regret - regret_ci,
            regret + regret_ci,
            color=color,
            alpha=0.12,
        )
        reward_axis.plot(steps, reward, color=color, label=label)
        reward_axis.fill_between(
            steps,
            reward - reward_ci,
            reward + reward_ci,
            color=color,
            alpha=0.12,
        )

    regret_axis.set(
        title="Best configuration per family: regret",
        xlabel="Step",
        ylabel="Cumulative pseudo-regret",
    )
    reward_axis.set(
        title="Best configuration per family: reward",
        xlabel="Step",
        ylabel="Running average reward",
    )
    if reference_reward is not None:
        reward_axis.axhline(reference_reward, color="black", linestyle=":", label="Best arm")
    for axis in (regret_axis, reward_axis):
        axis.legend(fontsize=8)


def _write_figure(
    comparison: ComparisonResult,
    scenario: Scenario,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    _plot_endpoint_rankings(comparison, axes[0, 0], axes[0, 1])
    _plot_best_family_curves(comparison, axes[1, 0], axes[1, 1], scenario.reference_reward)
    for axis in axes.flat:
        axis.grid(alpha=0.2)

    figure.suptitle(
        f"{scenario.title}: bandit hyperparameter sweep "
        f"({len(next(iter(comparison.runs.values())))} matched runs; 95% CIs)"
    )
    path = output_dir / "comparison.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _scenario_manifest(scenario: Scenario, comparison: ComparisonResult) -> dict:
    return {
        "title": scenario.title,
        "reference_reward": scenario.reference_reward,
        "grids": scenario.grids,
        "configurations": {
            name: {
                "family": family_of(name),
                "algorithm": runs[0].algorithm_name,
            }
            for name, runs in comparison.runs.items()
        },
    }


def write_results(
    results: dict[str, ComparisonResult],
    seeds: Sequence[int],
    n_steps: int,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Write the compact, reproducible result bundle for every scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "study.json"
    manifest_path.write_text(
        json.dumps(
            {
                "git_commit": _git_commit(),
                "n_steps": n_steps,
                "seeds": _seed_spec(seeds),
                "interval": "normal-approximation 95% confidence interval",
                "scenarios": {
                    scenario.key: _scenario_manifest(scenario, results[scenario.key])
                    for scenario in SCENARIOS
                },
            },
            indent=2,
        )
        + "\n"
    )

    paths: list[Path] = [manifest_path]
    for scenario in SCENARIOS:
        scenario_dir = output_dir / scenario.key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        comparison = results[scenario.key]
        paths.extend(
            [
                _write_summary(comparison, scenario_dir),
                _write_runs(comparison, seeds, scenario_dir),
                _write_paired(comparison, scenario_dir),
                _write_figure(comparison, scenario, scenario_dir),
            ]
        )
    return tuple(paths)


def main() -> None:
    results = run_study()
    paths = write_results(results, SEEDS, N_STEPS)

    for scenario in SCENARIOS:
        print(f"\n{scenario.title}")
        print(f"{'rank':>4} {'configuration':<36} {'reward':>10} {'regret':>10}")
        for rank, row in enumerate(_ranked(results[scenario.key]), start=1):
            print(f"{rank:>4} {row.name:<36} {row.mean_reward:>10.4f} {row.mean_regret:>10.2f}")
    print(f"\nSaved study to {paths[0].parent.resolve()}")


if __name__ == "__main__":
    main()
