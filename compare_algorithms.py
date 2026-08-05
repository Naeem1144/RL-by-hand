"""Compare tuned epsilon-greedy against UCB1 and Thompson sampling.

Benchmarks the constant-epsilon baseline, the best epsilon-greedy configuration
found in ``compare_hyperparameters.py`` (optimistic initialization +
``decay=0.90``), Thompson sampling, and a tuned UCB1 (``c=0.01``) on the same
Bernoulli bandits. Every configuration is averaged over 10 seeds for robust
results; prints a summary table and saves a comparison figure plus a CSV.

Note: UCB1 needs ``n_steps`` larger than ``n_arms`` to shine, because it must
pull every arm once before its confidence-based phase kicks in.
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
from algorithms.thompson import run_thompson
from algorithms.ucb import run_ucb


DECAY_RATE = 0.90
UCB_C = 0.01

N_ARMS_GRID = [10, 100, 1_000]
N_STEPS_GRID = [100, 1_000, 10_000]

# Algorithm runners: (label, callable(n_arms, n_steps, seed) -> results dict)
ALGORITHMS = {
    "epsgreedy const": lambda n_arms, n_steps, seed: run_bandit(
        n_arms=n_arms, n_steps=n_steps, epsilon=EPSILON, seed=seed
    ),
    "epsgreedy optimistic decay=0.90": lambda n_arms, n_steps, seed: run_bandit(
        n_arms=n_arms,
        n_steps=n_steps,
        epsilon=EPSILON,
        seed=seed,
        decay=True,
        decay_rate=DECAY_RATE,
        optimistic_initialization=True,
    ),
    "thompson sampling": lambda n_arms, n_steps, seed: run_thompson(
        n_arms=n_arms, n_steps=n_steps, seed=seed
    ),
    "ucb1 (c=0.01)": lambda n_arms, n_steps, seed: run_ucb(
        n_arms=n_arms, n_steps=n_steps, c=UCB_C, seed=seed
    ),
}


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"algorithm_comparison_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for the table
    rows: list[tuple[int, int, str, float, float, float, float]] = []
    # Store averaged series keyed by (n_arms, algorithm) for plotting
    series: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for n_arms in N_ARMS_GRID:
        for n_steps in N_STEPS_GRID:
            for label, run_fn in ALGORITHMS.items():
                def run_one(seed: int) -> dict[str, np.ndarray]:
                    return run_fn(n_arms, n_steps, seed)

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
                series[(n_arms, label)] = (steps, regret, reward_series)
                rows.append(
                    (n_arms, n_steps, label, avg_reward, std_reward, total_regret, std_regret)
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
    with csv_path.open("w") as handle:
        handle.write(
            "n_arms,n_steps,algorithm,"
            "final_avg_reward,final_avg_reward_std,total_regret,total_regret_std\n"
        )
        for n_arms, n_steps, label, avg_reward, std_reward, total_regret, std_regret in rows:
            handle.write(
                f"{n_arms},{n_steps},{label},"
                f"{avg_reward:.6f},{std_reward:.6f},{total_regret:.1f},{std_regret:.1f}\n"
            )

    # Comparison figure: regret (top) and average reward (bottom) vs steps
    labels = list(ALGORITHMS)
    colors = ["black", "tab:blue", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(2, len(N_ARMS_GRID), figsize=(16, 9), constrained_layout=True)

    for col, n_arms in enumerate(N_ARMS_GRID):
        regret_axis = axes[0, col]
        reward_axis = axes[1, col]

        for label, color in zip(labels, colors):
            steps, regret, reward_series = series[(n_arms, label)]
            linestyle = "--" if label == "epsgreedy const" else "-"
            regret_axis.plot(steps, regret, color=color, linestyle=linestyle, label=label)
            reward_axis.plot(steps, reward_series, color=color, linestyle=linestyle, label=label)

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

        regret_axis.set(title=f"{n_arms} arms", xlabel="Step", ylabel="Cumulative regret")
        reward_axis.set(xlabel="Step", ylabel="Average reward")
        reward_axis.set_ylim(0, 1.05)
        regret_axis.grid(alpha=0.25)
        reward_axis.grid(alpha=0.25)
        regret_axis.legend(fontsize=8)
        reward_axis.legend(fontsize=8)

    fig.suptitle(
        "Epsilon-greedy vs UCB1 vs Thompson sampling (averaged over "
        f"{len(SEEDS)} seeds, epsilon={EPSILON})",
        fontsize=15,
    )
    fig_path = Path("images/algorithm_comparison.png")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved summary to {out_dir.resolve()} and figure to {fig_path}")


if __name__ == "__main__":
    main()
