"""Epsilon-greedy Bernoulli multi-armed bandit, implemented from scratch."""

import numpy as np

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
) -> dict[str, np.ndarray]:
    """Run an epsilon-greedy Bernoulli multi-armed bandit simulation.

    On each step the agent explores a random arm with probability ``epsilon``
    and otherwise exploits its best current estimate. When ``decay_rate`` is
    provided, epsilon is multiplied by it after every step so exploration
    shrinks over time; when ``decay_rate`` is ``None`` epsilon stays constant.
    With ``optimistic_initialization`` all estimates start at 1.0 (the maximum
    reward), forcing early exploration of every arm.
    """
    rng = np.random.default_rng(seed)

    if optimistic_initialization:
        estimated_values = np.full(n_arms, 1.0)
    else:
        estimated_values = np.zeros(n_arms)

    total_pulls = np.zeros(n_arms, dtype=int)
    true_probs = rng.random(n_arms)

    # History
    rewards = np.zeros(n_steps, dtype=int)
    selected_arms = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        if rng.random() < epsilon:
            # Explore: pick a random arm
            selected_arm = rng.integers(n_arms)
        else:
            # Exploit: pick the arm with the highest estimate
            selected_arm = np.argmax(estimated_values)

        reward = int(rng.random() < true_probs[selected_arm])
        total_pulls[selected_arm] += 1
        estimated_values[selected_arm] += (
            reward - estimated_values[selected_arm]
        ) / total_pulls[selected_arm]

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
