"""Shared constants and helpers for the bandit comparison scripts."""

import numpy as np


# Comparison settings shared by both comparison scripts.
SEEDS = list(range(10))
EPSILON = 0.10
N_ARMS_GRID = [10, 100, 1_000]
N_STEPS_GRID = [100, 1_000, 10_000]


def summarize(results: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return steps, regret series, reward series, and final average reward."""
    rewards = results["rewards"]
    true_probs = results["true_probs"]
    selected_arms = results["selected_arms"]

    steps = np.arange(1, rewards.size + 1)
    cumulative_regret = np.cumsum(true_probs.max() - true_probs[selected_arms])
    running_average = np.cumsum(rewards) / steps

    return steps, cumulative_regret, running_average, float(running_average[-1])


def average_over_seeds(
    run_fn,
    seeds: list[int],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float, float, float, float, np.ndarray, np.ndarray
]:
    """Run ``run_fn(seed)`` for each seed and average the results.

    ``run_fn(seed)`` must return a results dict in ``run_bandit`` format.
    Returns (steps, mean regret series, mean reward series, mean final reward,
    std final reward, mean total regret, std total regret, per-seed final
    rewards, per-seed total regrets). Standard deviations use ``ddof=1`` so
    they can be turned into standard errors with ``std / sqrt(len(seeds))``.
    """
    regret_series: list[np.ndarray] = []
    reward_series: list[np.ndarray] = []
    final_rewards: list[float] = []
    total_regrets: list[float] = []
    steps: np.ndarray | None = None

    for seed in seeds:
        results = run_fn(seed)
        run_steps, regret, reward, avg_reward = summarize(results)
        if steps is None:
            steps = run_steps
        regret_series.append(regret)
        reward_series.append(reward)
        final_rewards.append(avg_reward)
        total_regrets.append(float(regret[-1]))

    if steps is None:
        raise ValueError("seeds must not be empty")

    return (
        steps,
        np.mean(regret_series, axis=0),
        np.mean(reward_series, axis=0),
        float(np.mean(final_rewards)),
        float(np.std(final_rewards, ddof=1)),
        float(np.mean(total_regrets)),
        float(np.std(total_regrets, ddof=1)),
        np.asarray(final_rewards),
        np.asarray(total_regrets),
    )
