import numpy as np

from visualizations import plot_bandit_results


SEED = 67
EPSILON = 0.10
N_ARMS = 1_000
N_STEPS = 1_000


def run_bandit(
    n_arms: int = N_ARMS,
    n_steps: int = N_STEPS,
    epsilon: float = EPSILON,
    seed: int = SEED,
    decay: float | None = None,
) -> dict[str, np.ndarray]:
    """Run an epsilon-greedy Bernoulli multi-armed bandit simulation.

    When ``decay`` is provided (e.g. 0.99), epsilon is multiplied by it after
    every step so exploration shrinks over time. When ``decay`` is ``None``,
    epsilon stays constant for the whole run.
    """
    rng = np.random.default_rng(seed)

    # bandit stats
    estimated_values = np.zeros(n_arms)
    total_pulls = np.zeros(n_arms, dtype=int)
    true_probs = rng.random(n_arms)

    # history
    rewards = np.zeros(n_steps, dtype=int)
    selected_arms = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        if rng.random() < epsilon:  # explore
            selected_arm = rng.integers(n_arms)
        else:  # exploit
            selected_arm = np.argmax(estimated_values)

        reward = int(rng.random() < true_probs[selected_arm])

        total_pulls[selected_arm] += 1

        estimated_values[selected_arm] += (
            reward - estimated_values[selected_arm]
        ) / total_pulls[selected_arm]

        rewards[step] = reward
        selected_arms[step] = selected_arm

        # decaying epsilon
        if decay is not None:
            epsilon *= decay

    return {
        "rewards": rewards,
        "selected_arms": selected_arms,
        "true_probs": true_probs,
        "estimated_values": estimated_values,
        "total_pulls": total_pulls,
    }


def main() -> None:
    results = run_bandit()
    output_path = plot_bandit_results(**results)
    print(f"Saved visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
