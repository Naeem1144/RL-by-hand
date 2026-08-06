"""Shared constants and helpers for the bandit comparison scripts."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Held-out instances shared by the two broad comparison scripts. Hyperparameter
# tuning and its confirmation set use separate ranges in
# ``compare_hyperparameters.py``.
BENCHMARK_ENVIRONMENT_SEEDS = list(range(20_000, 20_030))
EPSILON = 0.10
N_ARMS_GRID = [10, 100, 1_000]
N_STEPS_GRID = [100, 1_000, 10_000]


def generate_problem(n_arms: int, environment_seed: int) -> np.ndarray:
    """Generate one reproducible Uniform(0, 1) Bernoulli-bandit instance."""
    if isinstance(n_arms, bool) or not isinstance(n_arms, int):
        raise TypeError("n_arms must be an integer")
    if n_arms <= 0:
        raise ValueError("n_arms must be positive")
    if isinstance(environment_seed, bool) or not isinstance(environment_seed, int):
        raise TypeError("environment_seed must be an integer")
    if environment_seed < 0:
        raise ValueError("environment_seed must be non-negative")
    return np.random.default_rng(environment_seed).random(n_arms)


def simulation_seeds(environment_seed: int) -> tuple[int, int]:
    """Derive independent, reproducible policy and reward seeds for an instance."""
    sequence = np.random.SeedSequence([environment_seed, 0x524C])
    policy_sequence, reward_sequence = sequence.spawn(2)
    policy_seed = int(policy_sequence.generate_state(1, dtype=np.uint64)[0])
    reward_seed = int(reward_sequence.generate_state(1, dtype=np.uint64)[0])
    return policy_seed, reward_seed


def write_manifest(
    output_dir: Path,
    experiment: str,
    settings: dict[str, object],
    arm_mean_distribution: str = "independent Uniform(0, 1)",
) -> Path:
    """Write machine-readable provenance for a canonical result bundle."""
    source_files = sorted(Path(".").glob("*.py")) + sorted(Path("algorithms").glob("*.py"))
    source_files += [Path("pyproject.toml"), Path("uv.lock")]
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())

    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "arm_mean_distribution": arm_mean_distribution,
        "reward_distribution": "Bernoulli(true arm mean)",
        "settings": settings,
    }
    path = output_dir / "manifest.json"
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def summarize(results: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return steps, regret series, reward series, and final average reward."""
    rewards = np.asarray(results["rewards"])
    true_probs = np.asarray(results["true_probs"])
    selected_arms = np.asarray(results["selected_arms"])

    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError("rewards must be a non-empty one-dimensional array")
    if selected_arms.shape != rewards.shape:
        raise ValueError("selected_arms must have the same shape as rewards")
    if true_probs.ndim != 1 or true_probs.size == 0:
        raise ValueError("true_probs must be a non-empty one-dimensional array")
    if np.any((selected_arms < 0) | (selected_arms >= true_probs.size)):
        raise ValueError("selected_arms contains an out-of-range arm index")

    steps = np.arange(1, rewards.size + 1)
    cumulative_regret = np.cumsum(true_probs.max() - true_probs[selected_arms])
    running_average = np.cumsum(rewards) / steps

    return steps, cumulative_regret, running_average, float(running_average[-1])


def average_over_seeds(
    run_fn,
    seeds: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Run ``run_fn(seed)`` for each seed and average the results.

    ``run_fn(seed)`` must return a results dict in ``run_bandit`` format.
    Returns (steps, mean regret series, regret-series standard error, mean
    reward series, reward-series standard error, mean final reward, std final
    reward, mean total regret, std total regret, per-seed final rewards,
    per-seed total regrets). Standard deviations use ``ddof=1``.
    """
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required to estimate variability")

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

    regret_array = np.asarray(regret_series)
    reward_array = np.asarray(reward_series)
    root_n = np.sqrt(len(seeds))

    return (
        steps,
        np.mean(regret_array, axis=0),
        np.std(regret_array, axis=0, ddof=1) / root_n,
        np.mean(reward_array, axis=0),
        np.std(reward_array, axis=0, ddof=1) / root_n,
        float(np.mean(final_rewards)),
        float(np.std(final_rewards, ddof=1)),
        float(np.mean(total_regrets)),
        float(np.std(total_regrets, ddof=1)),
        np.asarray(final_rewards),
        np.asarray(total_regrets),
    )
