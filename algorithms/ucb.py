"""UCB1 (upper confidence bound) multi-armed bandit, implemented from scratch.

UCB1 balances exploration and exploitation by picking the arm that maximizes
``mean + c * sqrt(log(t) / n)``: the sample mean of observed rewards plus an
uncertainty bonus that shrinks as an arm is pulled more often. With
``c = sqrt(2)`` this is the classic UCB1 policy of Auer, Cesa-Bianchi &
Fischer (2002).
"""

import numpy as np

from algorithms._common import prepare_simulation, validate_nonnegative

SEED = 67
N_ARMS = 1_000
# UCB1 must pull every arm once before exploiting, so keep n_steps well above
# n_arms for the standalone demo (1,000-arm bootstrap + 9,000 UCB steps).
N_STEPS = 10_000
C = float(np.sqrt(2.0))


def run_ucb(
    n_arms: int = N_ARMS,
    n_steps: int = N_STEPS,
    c: float = C,
    seed: int = SEED,
    true_probs: np.ndarray | None = None,
    policy_seed: int | None = None,
    reward_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Run a UCB1 Bernoulli multi-armed bandit simulation.

    Each arm is pulled once first so its sample mean is defined; afterwards the
    arm with the highest upper-confidence index is chosen on every step. The
    returned dictionary matches the format of ``run_bandit`` so the same
    summarizers and plotters work for both algorithms.
    """
    c = validate_nonnegative(c, "c")
    inputs = prepare_simulation(n_arms, n_steps, seed, true_probs, policy_seed, reward_seed)
    reward_rng = inputs.reward_rng
    true_probs = inputs.true_probs

    estimated_values = np.zeros(n_arms)
    total_pulls = np.zeros(n_arms, dtype=int)
    rewards = np.zeros(n_steps, dtype=int)
    selected_arms = np.zeros(n_steps, dtype=int)

    # Bootstrap: pull each arm once so every sample mean is defined.
    # Capped by n_steps so runs with more arms than steps still terminate.
    bootstrap_steps = min(n_arms, n_steps)
    for step in range(bootstrap_steps):
        selected_arm = step
        reward = int(reward_rng.random() < true_probs[selected_arm])
        total_pulls[selected_arm] += 1
        estimated_values[selected_arm] += (reward - estimated_values[selected_arm]) / total_pulls[
            selected_arm
        ]
        rewards[step] = reward
        selected_arms[step] = selected_arm

    # UCB phase: pick the arm with the largest upper-confidence index.
    for step in range(bootstrap_steps, n_steps):
        t = step + 1  # 1-indexed step count
        ucb = estimated_values + c * np.sqrt(np.log(t) / total_pulls)
        selected_arm = int(np.argmax(ucb))
        reward = int(reward_rng.random() < true_probs[selected_arm])
        total_pulls[selected_arm] += 1
        estimated_values[selected_arm] += (reward - estimated_values[selected_arm]) / total_pulls[
            selected_arm
        ]
        rewards[step] = reward
        selected_arms[step] = selected_arm

    return {
        "rewards": rewards,
        "selected_arms": selected_arms,
        "true_probs": true_probs,
        "estimated_values": estimated_values,
        "total_pulls": total_pulls,
    }


def main() -> None:
    results = run_ucb()
    best_arm = int(np.argmax(results["true_probs"]))
    print(f"UCB1 over {N_STEPS} steps, {N_ARMS} arms, c={C:.4f}")
    print(f"mean reward:           {float(results['rewards'].mean()):.4f}")
    print(f"best arm probability:  {float(results['true_probs'].max()):.4f}")
    print(f"pulls of the best arm: {int(results['total_pulls'][best_arm])}")


if __name__ == "__main__":
    main()
