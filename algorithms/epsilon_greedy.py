"""Epsilon-greedy Bernoulli multi-armed bandit, implemented from scratch."""

import numpy as np

from algorithms._common import prepare_simulation, validate_probability
from visualizations import plot_bandit_results

SEED = 67
EPSILON = 0.10
N_ARMS = 1_000
N_STEPS = 1_000
OUTPUT_PATH = "images/bandit_results.png"


def run_bandit(
    n_arms: int = N_ARMS,
    n_steps: int = N_STEPS,
    epsilon: float = EPSILON,
    seed: int = SEED,
    decay: bool = False,
    decay_rate: float | None = None,
    optimistic_initialization: bool = False,
    true_probs: np.ndarray | None = None,
    policy_seed: int | None = None,
    reward_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Run an epsilon-greedy Bernoulli multi-armed bandit simulation.

    On each step the agent explores a random arm with probability ``epsilon``
    and otherwise exploits its best current estimate, breaking ties uniformly
    at random so equal estimates do not always favor the lowest-indexed arm.
    When ``decay`` is true, epsilon is multiplied by ``decay_rate`` after every
    step. With ``optimistic_initialization`` all estimates
    start at 1.0 (the maximum reward), which keeps untried arms attractive and
    encourages broad early exploration.

    ``true_probs`` and the two optional random-stream seeds support explicit,
    matched benchmark instances. When omitted, independent streams derived
    from ``seed`` generate the problem, policy choices, and rewards.
    """
    if not isinstance(decay, bool):
        raise TypeError("decay must be a boolean")
    if not isinstance(optimistic_initialization, bool):
        raise TypeError("optimistic_initialization must be a boolean")
    epsilon = validate_probability(epsilon, "epsilon")
    if decay:
        if decay_rate is None:
            raise ValueError("decay_rate is required when decay=True")
        decay_rate = validate_probability(decay_rate, "decay_rate")
        if decay_rate in (0.0, 1.0):
            raise ValueError("decay_rate must lie strictly between 0 and 1")
    elif decay_rate is not None:
        raise ValueError("decay_rate must be None when decay=False")

    inputs = prepare_simulation(n_arms, n_steps, seed, true_probs, policy_seed, reward_seed)
    policy_rng = inputs.policy_rng
    reward_rng = inputs.reward_rng
    true_probs = inputs.true_probs

    estimated_values = np.full(n_arms, 1.0) if optimistic_initialization else np.zeros(n_arms)

    total_pulls = np.zeros(n_arms, dtype=int)
    # History
    rewards = np.zeros(n_steps, dtype=int)
    selected_arms = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        if policy_rng.random() < epsilon:
            # Explore: pick a random arm
            selected_arm = policy_rng.integers(n_arms)
        else:
            # Exploit: pick the arm with the highest estimate. Ties are broken
            # uniformly at random so a block of equal estimates (e.g. every
            # zero-initialized arm at 0.0) does not always collapse onto arm 0.
            max_value = estimated_values.max()
            ties = np.flatnonzero(estimated_values == max_value)
            selected_arm = int(ties[0]) if ties.size == 1 else int(policy_rng.choice(ties))

        reward = int(reward_rng.random() < true_probs[selected_arm])
        total_pulls[selected_arm] += 1
        estimated_values[selected_arm] += (reward - estimated_values[selected_arm]) / total_pulls[
            selected_arm
        ]

        rewards[step] = reward
        selected_arms[step] = selected_arm

        if decay:
            epsilon *= decay_rate

    return {
        "rewards": rewards,
        "selected_arms": selected_arms,
        "true_probs": true_probs,
        "estimated_values": estimated_values,
        "total_pulls": total_pulls,
    }


def main() -> None:
    results = run_bandit()
    output_path = plot_bandit_results(**results, output_path=OUTPUT_PATH)
    print(f"Saved visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
