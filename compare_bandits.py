"""Compare constant vs decaying epsilon-greedy across decay rates.

Runs a constant-epsilon baseline and several exponential decay schedules on
the same seeded bandit environments over a grid of ``n_arms`` and ``n_steps``
values, prints a summary table, and saves a comparison figure plus a CSV.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from epsilon_greedy import run_bandit


SEED = 67
EPSILON = 0.10

N_ARMS_GRID = [10, 100, 1_000]
N_STEPS_GRID = [100, 1_000, 10_000]

# Exponential decay factors to sweep; None means constant epsilon.
DECAY_GRID = [0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

RUN_SPECS = [("constant", None)] + [
    (f"decay={decay}", decay) for decay in DECAY_GRID
]


def summarize(results: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return steps, regret series, reward series, and final average reward."""
    rewards = results["rewards"]
    true_probs = results["true_probs"]
    selected_arms = results["selected_arms"]

    steps = np.arange(1, rewards.size + 1)
    cumulative_regret = np.cumsum(true_probs.max() - true_probs[selected_arms])
    running_average = np.cumsum(rewards) / steps

    return steps, cumulative_regret, running_average, float(running_average[-1])


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"comparison_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for the table
    rows: list[tuple[int, int, str, float, float]] = []
    # Store full series keyed by (n_arms, label) for plotting
    series: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for n_arms in N_ARMS_GRID:
        for n_steps in N_STEPS_GRID:
            for label, decay in RUN_SPECS:
                results = run_bandit(
                    n_arms=n_arms,
                    n_steps=n_steps,
                    epsilon=EPSILON,
                    seed=SEED,
                    decay=decay,
                )
                steps, regret, reward_series, avg_reward = summarize(results)
                series[(n_arms, label)] = (steps, regret, reward_series)
                rows.append((n_arms, n_steps, label, avg_reward, float(regret[-1])))

    # Print summary table
    header = f"{'arms':>6} {'steps':>8} {'variant':<14} {'final avg':>10} {'total regret':>13}"
    print(header)
    print("-" * len(header))
    for n_arms, n_steps, label, avg_reward, regret in rows:
        print(
            f"{n_arms:>6} {n_steps:>8} {label:<14} {avg_reward:>10.4f} {regret:>13.1f}"
        )

    # Save CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w") as handle:
        handle.write("n_arms,n_steps,variant,final_avg_reward,total_regret\n")
        for n_arms, n_steps, label, avg_reward, regret in rows:
            handle.write(f"{n_arms},{n_steps},{label},{avg_reward:.6f},{regret:.1f}\n")

    # Comparison figure: regret (top) and average reward (bottom) vs steps
    fig, axes = plt.subplots(2, len(N_ARMS_GRID), figsize=(16, 9), constrained_layout=True)
    decay_values = [decay for _, decay in RUN_SPECS if decay is not None]
    cmap = plt.get_cmap("viridis")
    decay_colors = cmap(np.linspace(0.15, 0.9, len(decay_values)))

    for col, n_arms in enumerate(N_ARMS_GRID):
        regret_axis = axes[0, col]
        reward_axis = axes[1, col]

        # Constant-epsilon baseline (dashed black line)
        steps, regret, reward_series = series[(n_arms, "constant")]
        regret_axis.plot(
            steps, regret, color="black", linestyle="--", linewidth=2, label="constant"
        )
        reward_axis.plot(
            steps, reward_series, color="black", linestyle="--", linewidth=2, label="constant"
        )

        for decay, color in zip(decay_values, decay_colors):
            label = f"decay={decay}"
            steps, regret, reward_series = series[(n_arms, label)]
            regret_axis.plot(steps, regret, color=color, label=label)
            reward_axis.plot(steps, reward_series, color=color, label=label)

        best_prob = float(
            np.max(run_bandit(n_arms=n_arms, n_steps=1, seed=SEED)["true_probs"])
        )
        reward_axis.axhline(best_prob, color="tab:green", linestyle=":", label="Best arm")

        regret_axis.set(title=f"{n_arms} arms", xlabel="Step", ylabel="Cumulative regret")
        reward_axis.set(xlabel="Step", ylabel="Average reward")
        reward_axis.set_ylim(0, 1.05)
        regret_axis.grid(alpha=0.25)
        reward_axis.grid(alpha=0.25)
        regret_axis.legend(fontsize=7)
        reward_axis.legend(fontsize=7)

    fig.suptitle(
        "Epsilon-greedy decay sweep (seed "
        f"{SEED}, epsilon={EPSILON})",
        fontsize=15,
    )
    fig_path = out_dir / "comparison.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
