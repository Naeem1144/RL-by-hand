"""Regret-versus-horizon sweep on the fixed ten-arm Bernoulli problem.

Sweeps a focused subset of configurations over three horizons to show how
the ranking depends on the horizon: finite-horizon favourites (small UCB
scales, weak priors) versus asymptotic ones (the UCB1 constant), and how
quickly logarithmic strategies outpace epsilon-greedy.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from algorithms import UCB, EpsilonGreedy, ThompsonSampling
from evaluation import ComparisonResult, compare
from problems import BernoulliBandit
from study.compare import OUTPUT_DIR, PROBABILITIES, Z_95

HORIZONS = (1_000, 10_000, 100_000)
SEEDS = tuple(range(50))

CONFIGURATIONS = {
    "UCB c=0.5": UCB(c=0.5),
    "UCB c=1": UCB(c=1.0),
    "UCB c=1.41": UCB(c=float(np.sqrt(2.0))),
    "UCB c=2": UCB(c=2.0),
    "Thompson Beta(0.5,0.5)": ThompsonSampling(prior_alpha=0.5, prior_beta=0.5),
    "Epsilon epsilon=0.1": EpsilonGreedy(epsilon=0.1),
}


def _problem(seed: int):
    del seed
    return BernoulliBandit(PROBABILITIES, name="fixed ten-arm Bernoulli bandit")


def run_horizon_sweep(
    horizons: Sequence[int] = HORIZONS,
    seeds: Sequence[int] = SEEDS,
) -> dict[int, ComparisonResult]:
    """Evaluate the focused configuration subset at every horizon."""
    return {
        horizon: compare(CONFIGURATIONS, _problem, n_steps=horizon, seeds=seeds)
        for horizon in horizons
    }


def write_horizon_results(
    results: dict[int, ComparisonResult],
    seeds: Sequence[int],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Write one compact CSV: (horizon, configuration) rows with mean and risk stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "horizons.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "horizon",
                "configuration",
                "mean_reward",
                "reward_ci95_half_width",
                "mean_regret",
                "regret_ci95_half_width",
                "median_regret",
                "p90_regret",
                "max_regret",
                "regret_per_step",
            ]
        )
        for horizon in sorted(results):
            n_runs = len(seeds)
            for row in sorted(results[horizon].summaries(), key=lambda r: r.mean_regret):
                writer.writerow(
                    [
                        horizon,
                        row.name,
                        f"{row.mean_reward:.6f}",
                        f"{Z_95 * row.reward_std / np.sqrt(n_runs):.6f}",
                        f"{row.mean_regret:.6f}",
                        f"{Z_95 * row.regret_std / np.sqrt(n_runs):.6f}",
                        f"{row.median_regret:.6f}",
                        f"{row.p90_regret:.6f}",
                        f"{row.max_regret:.6f}",
                        f"{row.mean_regret / horizon:.6f}",
                    ]
                )
    return path


def main() -> None:
    results = run_horizon_sweep()
    path = write_horizon_results(results, SEEDS)
    for horizon in sorted(results):
        print(f"\nHorizon {horizon:,} steps")
        print(f"{'configuration':<22} {'mean_regret':>12} {'regret/step':>12} {'median':>9}")
        for row in sorted(results[horizon].summaries(), key=lambda r: r.mean_regret):
            print(
                f"{row.name:<22} {row.mean_regret:>12.1f} "
                f"{row.mean_regret / horizon:>12.5f} {row.median_regret:>9.1f}"
            )
    print(f"\nSaved horizon sweep to {path}")


if __name__ == "__main__":
    main()
