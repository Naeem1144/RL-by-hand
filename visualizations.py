"""Plotting helpers for the multi-armed bandit simulation."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_bandit_results(
    rewards: np.ndarray,
    selected_arms: np.ndarray,
    true_probs: np.ndarray,
    estimated_values: np.ndarray,
    total_pulls: np.ndarray,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Create a compact dashboard of learning and arm-selection behavior.

    When ``output_path`` is omitted, the figure is saved with a timestamped
    filename (e.g. ``bandit_results_20260803_153045.png``) so repeated runs
    do not overwrite each other.
    """
    rewards = np.asarray(rewards)
    selected_arms = np.asarray(selected_arms)
    true_probs = np.asarray(true_probs)
    estimated_values = np.asarray(estimated_values)
    total_pulls = np.asarray(total_pulls)

    if rewards.size == 0:
        raise ValueError("At least one reward is required to create plots.")
    if rewards.shape != selected_arms.shape:
        raise ValueError("rewards and selected_arms must have the same shape.")
    if not (true_probs.shape == estimated_values.shape == total_pulls.shape):
        raise ValueError("All arm-level arrays must have the same shape.")

    steps = np.arange(1, rewards.size + 1)
    optimal_arm = int(np.argmax(true_probs))
    optimal_probability = true_probs[optimal_arm]
    cumulative_reward = np.cumsum(rewards)
    running_average = cumulative_reward / steps
    cumulative_regret = np.cumsum(
        optimal_probability - true_probs[selected_arms]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Epsilon-greedy multi-armed bandit", fontsize=16)

    reward_axis = axes[0, 0]
    reward_axis.plot(steps, running_average, label="Observed average reward")
    reward_axis.axhline(
        optimal_probability,
        color="tab:green",
        linestyle="--",
        label="Best arm probability",
    )
    reward_axis.set(title="Reward over time", xlabel="Step", ylabel="Average reward")
    reward_axis.set_ylim(0, 1.05)
    reward_axis.legend()

    regret_axis = axes[0, 1]
    regret_axis.plot(steps, cumulative_regret, color="tab:red")
    regret_axis.set(
        title="Cumulative pseudo-regret",
        xlabel="Step",
        ylabel="Expected reward missed",
    )

    pulls_axis = axes[1, 0]
    top_count = min(20, total_pulls.size)
    top_arms = np.argsort(total_pulls)[-top_count:]
    bar_colors = [
        "tab:green" if arm == optimal_arm else "tab:blue" for arm in top_arms
    ]
    pulls_axis.barh([str(arm) for arm in top_arms], total_pulls[top_arms], color=bar_colors)
    pulls_axis.set(
        title=f"{top_count} most-selected arms",
        xlabel="Number of pulls",
        ylabel="Arm",
    )

    estimate_axis = axes[1, 1]
    pulled = total_pulls > 0
    point_sizes = 18 + 4 * np.sqrt(total_pulls[pulled])
    estimate_axis.scatter(
        true_probs[pulled],
        estimated_values[pulled],
        s=point_sizes,
        alpha=0.55,
        edgecolors="none",
    )
    estimate_axis.plot([0, 1], [0, 1], color="tab:gray", linestyle="--", label="Perfect estimate")
    estimate_axis.scatter(
        true_probs[optimal_arm],
        estimated_values[optimal_arm],
        marker="*",
        s=180,
        color="tab:green",
        label=f"Best arm ({optimal_arm})",
        zorder=3,
    )
    estimate_axis.set(
        title="Learned estimates for pulled arms",
        xlabel="True reward probability",
        ylabel="Estimated reward probability",
        xlim=(0, 1.05),
        ylim=(0, 1.05),
    )
    estimate_axis.legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"bandit_results_{timestamp}.png"

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return destination
