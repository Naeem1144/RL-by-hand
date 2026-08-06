"""Evaluate tuned policies across several Bernoulli arm-mean distributions."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bandit_utils import average_over_seeds, simulation_seeds, write_manifest
from compare_algorithms import ALGORITHMS

N_ARMS = 100
N_STEPS = 2_000
ENVIRONMENT_SEEDS = list(range(30_000, 30_030))
T_CRIT_95 = 2.0452


def generate_suite_problem(suite: str, environment_seed: int) -> np.ndarray:
    """Generate a reproducible problem from a named arm-mean suite."""
    rng = np.random.default_rng(environment_seed)
    if suite == "uniform":
        return rng.random(N_ARMS)
    if suite == "small-gap":
        probabilities = rng.uniform(0.49, 0.51, N_ARMS)
        probabilities[rng.integers(N_ARMS)] = 0.515
        return probabilities
    if suite == "clustered":
        high_cluster = rng.random(N_ARMS) < 0.20
        probabilities = np.where(
            high_cluster,
            rng.normal(0.75, 0.025, N_ARMS),
            rng.normal(0.25, 0.05, N_ARMS),
        )
        return np.clip(probabilities, 0.0, 1.0)
    if suite == "rare-good":
        probabilities = rng.uniform(0.10, 0.30, N_ARMS)
        probabilities[rng.integers(N_ARMS)] = 0.80
        return probabilities
    raise ValueError(f"unknown problem suite: {suite}")


PROBLEM_SUITES = ("uniform", "small-gap", "clustered", "rare-good")


def main() -> None:
    output_dir = Path("results/problem_suites")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, float, float, float, float]] = []
    per_instance: list[tuple[str, str, int, float, float]] = []

    for suite in PROBLEM_SUITES:
        for label, run_fn in ALGORITHMS.items():

            def run_one(
                environment_seed: int,
                suite: str = suite,
                run_fn=run_fn,
            ) -> dict[str, np.ndarray]:
                true_probs = generate_suite_problem(suite, environment_seed)
                policy_seed, reward_seed = simulation_seeds(environment_seed)
                return run_fn(N_ARMS, N_STEPS, true_probs, policy_seed, reward_seed)

            (
                _,
                _,
                _,
                _,
                _,
                avg_reward,
                std_reward,
                total_regret,
                std_regret,
                per_rewards,
                per_regrets,
            ) = average_over_seeds(run_one, ENVIRONMENT_SEEDS)
            rows.append((suite, label, avg_reward, std_reward, total_regret, std_regret))
            per_instance.extend(
                (suite, label, seed, float(reward), float(regret))
                for seed, reward, regret in zip(
                    ENVIRONMENT_SEEDS, per_rewards, per_regrets, strict=True
                )
            )

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "suite",
                "algorithm",
                "avg_reward",
                "std_reward",
                "total_regret",
                "std_regret",
            ]
        )
        writer.writerows(rows)

    with (output_dir / "per_instance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "suite",
                "algorithm",
                "environment_seed",
                "final_avg_reward",
                "total_regret",
            ]
        )
        writer.writerows(per_instance)

    labels = list(ALGORITHMS)
    colors = ["#111827", "#2563EB", "#F97316", "#DC2626"]
    x = np.arange(len(PROBLEM_SUITES))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        selected = [row for row in rows if row[1] == label]
        positions = x + (index - 1.5) * width
        axes[0].bar(
            positions,
            [row[4] for row in selected],
            width,
            yerr=[T_CRIT_95 * row[5] / np.sqrt(len(ENVIRONMENT_SEEDS)) for row in selected],
            capsize=3,
            color=color,
            label=label,
        )
        axes[1].bar(
            positions,
            [row[2] for row in selected],
            width,
            yerr=[T_CRIT_95 * row[3] / np.sqrt(len(ENVIRONMENT_SEEDS)) for row in selected],
            capsize=3,
            color=color,
            label=label,
        )

    axes[0].set(ylabel="Mean cumulative pseudo-regret", title="Regret across problem suites")
    axes[1].set(ylabel="Mean final average reward", title="Reward across problem suites")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xticks(x, PROBLEM_SUITES)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"Distribution robustness · K={N_ARMS}, T={N_STEPS:,}, "
        f"{len(ENVIRONMENT_SEEDS)} held-out instances; 95% CIs"
    )
    figure_path = Path("images/problem_suite_comparison.png")
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    write_manifest(
        output_dir,
        "arm-mean distribution robustness",
        {
            "n_arms": N_ARMS,
            "n_steps": N_STEPS,
            "environment_seeds": ENVIRONMENT_SEEDS,
            "problem_suites": list(PROBLEM_SUITES),
            "algorithms": labels,
        },
        arm_mean_distribution="named suite; see compare_problem_suites.py",
    )
    print(f"Saved results to {output_dir.resolve()} and figure to {figure_path}")


if __name__ == "__main__":
    main()
