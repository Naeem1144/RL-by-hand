"""Compare constant vs decaying epsilon-greedy across decay rates.

Runs a constant-epsilon baseline and several exponential decay schedules, each
with and without optimistic initialization, over a grid of ``n_arms`` and
``n_steps`` values. Every configuration is averaged over 30 held-out matched
instances; the script saves aggregate and per-instance data plus a figure.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.epsilon_greedy import run_bandit
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

# Exponential decay factors to sweep; None means constant epsilon.
DECAY_GRID = [0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

OPTIMISTIC_INITIALIZATION = [True, False]
PLOT_HORIZON = max(N_STEPS_GRID)
T_CRIT_95 = 2.0452  # Student's t, 29 degrees of freedom.

RUN_SPECS = [("constant", None)] + [(f"decay={decay}", decay) for decay in DECAY_GRID]


def main() -> None:
    out_dir = Path("results/epsilon_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for the table
    rows: list[tuple[int, int, bool, str, float, float, float, float]] = []
    # Keep the horizon in the key so grid results cannot overwrite one another.
    series: dict[
        tuple[int, int, bool, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    per_seed_rows: list[tuple[int, int, bool, str, int, float, float]] = []

    for n_arms in N_ARMS_GRID:
        for n_steps in N_STEPS_GRID:
            for opt_init in OPTIMISTIC_INITIALIZATION:
                for label, decay in RUN_SPECS:

                    def run_one(
                        environment_seed: int,
                        n_arms: int = n_arms,
                        n_steps: int = n_steps,
                        decay: float | None = decay,
                        opt_init: bool = opt_init,
                    ) -> dict[str, np.ndarray]:
                        true_probs = generate_problem(n_arms, environment_seed)
                        policy_seed, reward_seed = simulation_seeds(environment_seed)
                        return run_bandit(
                            n_arms=n_arms,
                            n_steps=n_steps,
                            epsilon=EPSILON,
                            true_probs=true_probs,
                            policy_seed=policy_seed,
                            reward_seed=reward_seed,
                            decay=decay is not None,
                            decay_rate=decay,
                            optimistic_initialization=opt_init,
                        )

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
                    series[(n_arms, n_steps, opt_init, label)] = (
                        steps,
                        regret,
                        regret_se,
                        reward_series,
                        reward_se,
                    )
                    rows.append(
                        (
                            n_arms,
                            n_steps,
                            opt_init,
                            label,
                            avg_reward,
                            std_reward,
                            total_regret,
                            std_regret,
                        )
                    )
                    per_seed_rows.extend(
                        (
                            n_arms,
                            n_steps,
                            opt_init,
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
        f"{'arms':>6} {'steps':>8} {'init':<10} {'variant':<14} "
        f"{'final avg':>10} {'rew std':>8} {'total regret':>13} {'reg std':>9}"
    )
    print(header)
    print("-" * len(header))
    for n_arms, n_steps, opt_init, label, avg_reward, std_reward, total_regret, std_regret in rows:
        init = "optimistic" if opt_init else "zero-init"
        print(
            f"{n_arms:>6} {n_steps:>8} {init:<10} {label:<14} "
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
                "optimistic_init",
                "variant",
                "final_avg_reward",
                "final_avg_reward_std",
                "total_regret",
                "total_regret_std",
            ]
        )
        for (
            n_arms,
            n_steps,
            opt_init,
            label,
            avg_reward,
            std_reward,
            total_regret,
            std_regret,
        ) in rows:
            writer.writerow(
                [
                    n_arms,
                    n_steps,
                    int(opt_init),
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
                "optimistic_init",
                "variant",
                "environment_seed",
                "final_avg_reward",
                "total_regret",
            ]
        )
        writer.writerows(per_seed_rows)

    write_manifest(
        out_dir,
        "epsilon-greedy schedule sweep",
        {
            "environment_seeds": BENCHMARK_ENVIRONMENT_SEEDS,
            "n_arms_grid": N_ARMS_GRID,
            "n_steps_grid": N_STEPS_GRID,
            "epsilon": EPSILON,
            "decay_grid": DECAY_GRID,
            "plot_horizon": PLOT_HORIZON,
        },
    )

    # Comparison figure: regret (top) and average reward (bottom) vs steps,
    # with separate row blocks for zero-initialized and optimistic estimates.
    init_labels = {True: "optimistic", False: "zero-init"}
    fig, axes = plt.subplots(
        2 * len(OPTIMISTIC_INITIALIZATION),
        len(N_ARMS_GRID),
        figsize=(16, 12),
        constrained_layout=True,
    )
    decay_values = [decay for _, decay in RUN_SPECS if decay is not None]
    cmap = plt.get_cmap("viridis")
    decay_colors = cmap(np.linspace(0.15, 0.9, len(decay_values)))

    for init_row, opt_init in enumerate(OPTIMISTIC_INITIALIZATION):
        for col, n_arms in enumerate(N_ARMS_GRID):
            regret_axis = axes[2 * init_row, col]
            reward_axis = axes[2 * init_row + 1, col]

            # Constant-epsilon baseline (dashed black line)
            steps, regret, regret_se, reward_series, reward_se = series[
                (n_arms, PLOT_HORIZON, opt_init, "constant")
            ]
            regret_axis.plot(
                steps, regret, color="black", linestyle="--", linewidth=2, label="constant"
            )
            reward_axis.plot(
                steps, reward_series, color="black", linestyle="--", linewidth=2, label="constant"
            )
            regret_axis.fill_between(
                steps,
                regret - T_CRIT_95 * regret_se,
                regret + T_CRIT_95 * regret_se,
                color="black",
                alpha=0.08,
            )
            reward_axis.fill_between(
                steps,
                reward_series - T_CRIT_95 * reward_se,
                reward_series + T_CRIT_95 * reward_se,
                color="black",
                alpha=0.08,
            )

            for decay, color in zip(decay_values, decay_colors, strict=True):
                label = f"decay={decay}"
                steps, regret, regret_se, reward_series, reward_se = series[
                    (n_arms, PLOT_HORIZON, opt_init, label)
                ]
                regret_axis.plot(steps, regret, color=color, label=label)
                reward_axis.plot(steps, reward_series, color=color, label=label)
                regret_axis.fill_between(
                    steps,
                    regret - T_CRIT_95 * regret_se,
                    regret + T_CRIT_95 * regret_se,
                    color=color,
                    alpha=0.06,
                )
                reward_axis.fill_between(
                    steps,
                    reward_series - T_CRIT_95 * reward_se,
                    reward_series + T_CRIT_95 * reward_se,
                    color=color,
                    alpha=0.06,
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

            title = f"{n_arms} arms · {init_labels[opt_init]}"
            regret_axis.set(title=title, xlabel="Step", ylabel="Cumulative regret")
            reward_axis.set(xlabel="Step", ylabel="Average reward")
            reward_axis.set_ylim(0, 1.05)
            regret_axis.grid(alpha=0.25)
            reward_axis.grid(alpha=0.25)
            regret_axis.legend(fontsize=7)
            reward_axis.legend(fontsize=7)

    fig.suptitle(
        "Epsilon-greedy decay sweep "
        f"(T={PLOT_HORIZON:,}, {len(BENCHMARK_ENVIRONMENT_SEEDS)} held-out instances; "
        "shaded 95% CIs)",
        fontsize=15,
    )
    fig_path = Path("images/comparison.png")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved summary to {out_dir.resolve()} and figure to {fig_path}")


if __name__ == "__main__":
    main()
