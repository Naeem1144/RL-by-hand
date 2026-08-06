"""Compare tuned epsilon-greedy against UCB(c) and Thompson sampling.

Benchmarks the constant-epsilon baseline, the best epsilon-greedy configuration
selected in ``compare_hyperparameters.py``: optimistic epsilon-greedy with
``decay=0.90``, Thompson sampling with ``Beta(5,5)``, and UCB(c) with
``c=0.003``. The constant-epsilon policy remains as an untuned baseline. All use the same
held-out Bernoulli bandits. Every configuration uses explicit matched problem,
policy, and reward streams; the tuning seeds are never reused here.

Note: this UCB implementation needs ``n_steps`` larger than ``n_arms`` to
adapt, because it must
pull every arm once before its confidence-based phase kicks in.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.epsilon_greedy import run_bandit
from algorithms.thompson import run_thompson
from algorithms.ucb import run_ucb
from bandit_utils import (
    BENCHMARK_ENVIRONMENT_SEEDS,
    EPSILON,
    N_ARMS_GRID,
    N_STEPS_GRID,
    average_over_seeds,
    generate_problem,
    simulation_seeds,
    write_manifest,
)

DECAY_RATE = 0.90
UCB_C = 0.003
THOMPSON_PRIOR = 5.0
PLOT_HORIZON = max(N_STEPS_GRID)
T_CRIT_95 = 2.0452  # Student's t, 29 degrees of freedom.


def run_constant_epsilon(
    n_arms: int,
    n_steps: int,
    true_probs: np.ndarray,
    policy_seed: int,
    reward_seed: int,
) -> dict[str, np.ndarray]:
    return run_bandit(
        n_arms=n_arms,
        n_steps=n_steps,
        epsilon=EPSILON,
        true_probs=true_probs,
        policy_seed=policy_seed,
        reward_seed=reward_seed,
    )


def run_optimistic_decay(
    n_arms: int,
    n_steps: int,
    true_probs: np.ndarray,
    policy_seed: int,
    reward_seed: int,
) -> dict[str, np.ndarray]:
    return run_bandit(
        n_arms=n_arms,
        n_steps=n_steps,
        epsilon=EPSILON,
        true_probs=true_probs,
        policy_seed=policy_seed,
        reward_seed=reward_seed,
        decay=True,
        decay_rate=DECAY_RATE,
        optimistic_initialization=True,
    )


def run_tuned_thompson(
    n_arms: int,
    n_steps: int,
    true_probs: np.ndarray,
    policy_seed: int,
    reward_seed: int,
) -> dict[str, np.ndarray]:
    return run_thompson(
        n_arms=n_arms,
        n_steps=n_steps,
        true_probs=true_probs,
        policy_seed=policy_seed,
        reward_seed=reward_seed,
        prior_alpha=THOMPSON_PRIOR,
        prior_beta=THOMPSON_PRIOR,
    )


def run_tuned_ucb(
    n_arms: int,
    n_steps: int,
    true_probs: np.ndarray,
    policy_seed: int,
    reward_seed: int,
) -> dict[str, np.ndarray]:
    return run_ucb(
        n_arms=n_arms,
        n_steps=n_steps,
        c=UCB_C,
        true_probs=true_probs,
        policy_seed=policy_seed,
        reward_seed=reward_seed,
    )


ALGORITHMS = {
    "epsgreedy const": run_constant_epsilon,
    "epsgreedy optimistic decay=0.90": run_optimistic_decay,
    "thompson Beta(5,5)": run_tuned_thompson,
    "ucb(c=0.003)": run_tuned_ucb,
}


def main() -> None:
    out_dir = Path("results/algorithm_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for the table
    rows: list[tuple[int, int, str, float, float, float, float]] = []
    # Keep the horizon in the key so grid results cannot overwrite one another.
    series: dict[
        tuple[int, int, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    per_seed_rows: list[tuple[int, int, str, int, float, float]] = []

    for n_arms in N_ARMS_GRID:
        for n_steps in N_STEPS_GRID:
            for label, run_fn in ALGORITHMS.items():

                def run_one(
                    environment_seed: int,
                    n_arms: int = n_arms,
                    n_steps: int = n_steps,
                    run_fn=run_fn,
                ) -> dict[str, np.ndarray]:
                    true_probs = generate_problem(n_arms, environment_seed)
                    policy_seed, reward_seed = simulation_seeds(environment_seed)
                    return run_fn(n_arms, n_steps, true_probs, policy_seed, reward_seed)

                (
                    steps,
                    regret,
                    regret_se,
                    reward_series,
                    reward_se,
                    avg_reward,
                    std_reward,
                    total_regret,
                    std_regret,
                    per_seed_rewards,
                    per_seed_regrets,
                ) = average_over_seeds(run_one, BENCHMARK_ENVIRONMENT_SEEDS)
                series[(n_arms, n_steps, label)] = (
                    steps,
                    regret,
                    regret_se,
                    reward_series,
                    reward_se,
                )
                rows.append(
                    (n_arms, n_steps, label, avg_reward, std_reward, total_regret, std_regret)
                )
                per_seed_rows.extend(
                    (
                        n_arms,
                        n_steps,
                        label,
                        environment_seed,
                        float(seed_reward),
                        float(seed_regret),
                    )
                    for environment_seed, seed_reward, seed_regret in zip(
                        BENCHMARK_ENVIRONMENT_SEEDS,
                        per_seed_rewards,
                        per_seed_regrets,
                        strict=True,
                    )
                )

    # Print summary table
    header = (
        f"{'arms':>6} {'steps':>8} {'algorithm':<28} "
        f"{'final avg':>10} {'rew std':>8} {'total regret':>13} {'reg std':>9}"
    )
    print(header)
    print("-" * len(header))
    for n_arms, n_steps, label, avg_reward, std_reward, total_regret, std_regret in rows:
        print(
            f"{n_arms:>6} {n_steps:>8} {label:<28} "
            f"{avg_reward:>10.4f} {std_reward:>8.4f} "
            f"{total_regret:>13.1f} {std_regret:>9.1f}"
        )

    # Save CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "n_arms",
                "n_steps",
                "algorithm",
                "final_avg_reward",
                "final_avg_reward_std",
                "total_regret",
                "total_regret_std",
            ]
        )
        for n_arms, n_steps, label, avg_reward, std_reward, total_regret, std_regret in rows:
            writer.writerow(
                [
                    n_arms,
                    n_steps,
                    label,
                    f"{avg_reward:.6f}",
                    f"{std_reward:.6f}",
                    f"{total_regret:.6f}",
                    f"{std_regret:.6f}",
                ]
            )

    with (out_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "n_arms",
                "n_steps",
                "algorithm",
                "environment_seed",
                "final_avg_reward",
                "total_regret",
            ]
        )
        writer.writerows(per_seed_rows)

    write_manifest(
        out_dir,
        "cross-algorithm comparison",
        {
            "environment_seeds": BENCHMARK_ENVIRONMENT_SEEDS,
            "n_arms_grid": N_ARMS_GRID,
            "n_steps_grid": N_STEPS_GRID,
            "algorithms": list(ALGORITHMS),
            "plot_horizon": PLOT_HORIZON,
        },
    )

    # Comparison figure: regret (top) and average reward (bottom) vs steps
    labels = list(ALGORITHMS)
    colors = ["black", "tab:blue", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(2, len(N_ARMS_GRID), figsize=(16, 9), constrained_layout=True)

    for col, n_arms in enumerate(N_ARMS_GRID):
        regret_axis = axes[0, col]
        reward_axis = axes[1, col]

        for label, color in zip(labels, colors, strict=True):
            steps, regret, regret_se, reward_series, reward_se = series[
                (n_arms, PLOT_HORIZON, label)
            ]
            linestyle = "--" if label == "epsgreedy const" else "-"
            regret_axis.plot(steps, regret, color=color, linestyle=linestyle, label=label)
            reward_axis.plot(steps, reward_series, color=color, linestyle=linestyle, label=label)
            regret_axis.fill_between(
                steps,
                regret - T_CRIT_95 * regret_se,
                regret + T_CRIT_95 * regret_se,
                color=color,
                alpha=0.10,
            )
            reward_axis.fill_between(
                steps,
                reward_series - T_CRIT_95 * reward_se,
                reward_series + T_CRIT_95 * reward_se,
                color=color,
                alpha=0.10,
            )

        best_prob = float(
            np.mean(
                [
                    np.max(generate_problem(n_arms, environment_seed))
                    for environment_seed in BENCHMARK_ENVIRONMENT_SEEDS
                ]
            )
        )
        reward_axis.axhline(best_prob, color="tab:green", linestyle=":", label="Best arm")

        regret_axis.set(title=f"{n_arms} arms", xlabel="Step", ylabel="Cumulative regret")
        reward_axis.set(xlabel="Step", ylabel="Average reward")
        reward_axis.set_ylim(0, 1.05)
        regret_axis.grid(alpha=0.25)
        reward_axis.grid(alpha=0.25)
        regret_axis.legend(fontsize=8)
        reward_axis.legend(fontsize=8)

    fig.suptitle(
        "Epsilon-greedy vs UCB(c) vs Thompson sampling "
        f"(T={PLOT_HORIZON:,}, {len(BENCHMARK_ENVIRONMENT_SEEDS)} held-out "
        "instances; shaded 95% CIs)",
        fontsize=15,
    )
    fig_path = Path("images/algorithm_comparison.png")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved summary to {out_dir.resolve()} and figure to {fig_path}")


if __name__ == "__main__":
    main()
