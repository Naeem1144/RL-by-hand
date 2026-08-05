"""Compare constant vs decaying epsilon-greedy across decay rates.

Runs a constant-epsilon baseline and several exponential decay schedules, each
with and without optimistic initialization, over a grid of ``n_arms`` and
``n_steps`` values. Every configuration is averaged over 10 seeds for robust
results; prints a summary table and saves a comparison figure plus a CSV.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bandit_utils import (
    EPSILON,
    N_ARMS_GRID,
    N_STEPS_GRID,
    SEEDS,
    average_over_seeds,
)
from algorithms.epsilon_greedy import run_bandit


# Exponential decay factors to sweep; None means constant epsilon.
DECAY_GRID = [0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

OPTIMISTIC_INITIALIZATION = [True, False]

RUN_SPECS = [("constant", None)] + [
    (f"decay={decay}", decay) for decay in DECAY_GRID
]


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"comparison_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for the table
    rows: list[tuple[int, int, bool, str, float, float, float, float]] = []
    # Store averaged series keyed by (n_arms, optimistic_init, label) for plotting
    series: dict[tuple[int, bool, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for n_arms in N_ARMS_GRID:
        for n_steps in N_STEPS_GRID:
            for opt_init in OPTIMISTIC_INITIALIZATION:
                for label, decay in RUN_SPECS:
                    def run_one(seed: int) -> dict[str, np.ndarray]:
                        return run_bandit(
                            n_arms=n_arms,
                            n_steps=n_steps,
                            epsilon=EPSILON,
                            seed=seed,
                            decay=decay is not None,
                            decay_rate=decay,
                            optimistic_initialization=opt_init,
                        )

                    (
                        steps,
                        regret,
                        reward_series,
                        avg_reward,
                        std_reward,
                        total_regret,
                        std_regret,
                        _,
                        _,
                    ) = average_over_seeds(run_one, SEEDS)
                    series[(n_arms, opt_init, label)] = (steps, regret, reward_series)
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
    with csv_path.open("w") as handle:
        handle.write(
            "n_arms,n_steps,optimistic_init,variant,"
            "final_avg_reward,final_avg_reward_std,total_regret,total_regret_std\n"
        )
        for n_arms, n_steps, opt_init, label, avg_reward, std_reward, total_regret, std_regret in rows:
            handle.write(
                f"{n_arms},{n_steps},{int(opt_init)},{label},"
                f"{avg_reward:.6f},{std_reward:.6f},{total_regret:.1f},{std_regret:.1f}\n"
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
            steps, regret, reward_series = series[(n_arms, opt_init, "constant")]
            regret_axis.plot(
                steps, regret, color="black", linestyle="--", linewidth=2, label="constant"
            )
            reward_axis.plot(
                steps, reward_series, color="black", linestyle="--", linewidth=2, label="constant"
            )

            for decay, color in zip(decay_values, decay_colors):
                label = f"decay={decay}"
                steps, regret, reward_series = series[(n_arms, opt_init, label)]
                regret_axis.plot(steps, regret, color=color, label=label)
                reward_axis.plot(steps, reward_series, color=color, label=label)

            best_prob = float(
                np.mean(
                    [
                        np.max(
                            run_bandit(n_arms=n_arms, n_steps=1, seed=seed)["true_probs"]
                        )
                        for seed in SEEDS
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
        "Epsilon-greedy decay sweep (averaged over "
        f"{len(SEEDS)} seeds, epsilon={EPSILON}, with/without optimistic initialization)",
        fontsize=15,
    )
    fig_path = Path("images/comparison.png")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved summary to {out_dir.resolve()} and figure to {fig_path}")


if __name__ == "__main__":
    main()
