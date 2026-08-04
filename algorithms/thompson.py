"""Thompson sampling for a Bernoulli multi-armed bandit, from scratch.

Each arm is modelled with a Beta posterior: ``Beta(successes + 1, failures + 1)``
starting from a flat ``Beta(1, 1)`` prior. On every step a reward probability is
sampled from each arm's posterior and the arm with the highest sample is pulled,
which automatically balances exploration and exploitation.
"""

import numpy as np


SEED = 67
N_ARMS = 1_000
N_STEPS = 10_000


def run_thompson(
    n_arms: int = N_ARMS,
    n_steps: int = N_STEPS,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Run a Beta-Bernoulli Thompson-sampling bandit simulation.

    Returns the same dictionary format as ``run_bandit`` so the shared
    summarizers and plotters work unchanged.
    """
    rng = np.random.default_rng(seed)

    # Beta(1, 1) prior counts: successes and failures for each arm
    successes = np.zeros(n_arms, dtype=int)
    failures = np.zeros(n_arms, dtype=int)
    total_pulls = np.zeros(n_arms, dtype=int)
    true_probs = rng.random(n_arms)

    rewards = np.zeros(n_steps, dtype=int)
    selected_arms = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        # Sample a plausible reward probability from each arm's posterior
        samples = rng.beta(successes + 1, failures + 1)
        selected_arm = int(np.argmax(samples))

        reward = int(rng.random() < true_probs[selected_arm])
        successes[selected_arm] += reward
        failures[selected_arm] += 1 - reward
        total_pulls[selected_arm] += 1

        rewards[step] = reward
        selected_arms[step] = selected_arm

    # Posterior means double as estimated values
    estimated_values = successes / np.maximum(total_pulls, 1)

    return {
        "rewards": rewards,
        "selected_arms": selected_arms,
        "true_probs": true_probs,
        "estimated_values": estimated_values,
        "total_pulls": total_pulls,
    }


def main() -> None:
    results = run_thompson()
    best_arm = int(np.argmax(results["true_probs"]))
    print(f"Thompson sampling over {N_STEPS} steps, {N_ARMS} arms")
    print(f"mean reward:           {float(results['rewards'].mean()):.4f}")
    print(f"best arm probability:  {float(results['true_probs'].max()):.4f}")
    print(f"pulls of the best arm: {int(results['total_pulls'][best_arm])}")


if __name__ == "__main__":
    main()
