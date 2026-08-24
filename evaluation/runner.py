"""The single interaction loop shared by every algorithm and problem."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from algorithms.base import BanditAlgorithm
from problems.base import BanditProblem


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _seed_sequence(seed: int, name: str) -> np.random.SeedSequence:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(seed) < 0:
        raise ValueError(f"{name} must be non-negative")
    return np.random.SeedSequence(int(seed))


@dataclass(frozen=True)
class RunResult:
    """Immutable history and metrics from one algorithm/problem run."""

    algorithm_name: str
    problem_name: str
    rewards: np.ndarray
    selected_arms: np.ndarray
    expected_reward_of_selected: np.ndarray
    expected_reward_of_best: np.ndarray
    estimated_values: np.ndarray
    total_pulls: np.ndarray

    @property
    def steps(self) -> np.ndarray:
        return np.arange(1, self.rewards.size + 1)

    @property
    def running_average_reward(self) -> np.ndarray:
        return np.cumsum(self.rewards) / self.steps

    @property
    def cumulative_regret(self) -> np.ndarray:
        losses = self.expected_reward_of_best - self.expected_reward_of_selected
        return np.cumsum(losses)

    @property
    def average_reward(self) -> float:
        return float(self.rewards.mean())

    @property
    def total_regret(self) -> float:
        return float(self.cumulative_regret[-1])


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array).copy()
    result.setflags(write=False)
    return result


def run(
    algorithm: BanditAlgorithm,
    problem: BanditProblem,
    n_steps: int,
    *,
    seed: int = 0,
) -> RunResult:
    """Run one algorithm on one problem.

    The algorithm is reset before the run.  Policy and reward randomness use
    independent streams so two algorithms can receive matched reward streams
    without having to consume policy randomness in the same way.

    The problem is not reset, so pass a fresh instance of a mutating problem
    for each run; a single ``run`` on one fresh instance is always safe.

    The expected reward of the selected and best arm is recorded after every
    sample, so regret stays correct for problems whose arm means change over
    time.
    """
    n_steps = _positive_integer(n_steps, "n_steps")
    n_arms = _positive_integer(problem.n_arms, "problem.n_arms")
    expected_rewards = np.asarray(problem.expected_rewards, dtype=float)
    if expected_rewards.shape != (n_arms,):
        raise ValueError(f"problem.expected_rewards must have shape ({n_arms},)")
    if not np.all(np.isfinite(expected_rewards)):
        raise ValueError("problem.expected_rewards must be finite")

    master = _seed_sequence(seed, "seed")
    policy_sequence, reward_sequence = master.spawn(2)
    policy_rng = np.random.default_rng(policy_sequence)
    reward_rng = np.random.default_rng(reward_sequence)

    algorithm.reset(n_arms, policy_rng)
    rewards = np.zeros(n_steps, dtype=float)
    selected_arms = np.zeros(n_steps, dtype=int)
    expected_of_selected = np.zeros(n_steps, dtype=float)
    expected_of_best = np.zeros(n_steps, dtype=float)

    for step in range(n_steps):
        arm = algorithm.select_arm(step)
        if isinstance(arm, bool) or not isinstance(arm, Integral):
            raise TypeError(f"{algorithm.name}.select_arm() must return an integer")
        arm = int(arm)
        if not 0 <= arm < n_arms:
            raise ValueError(
                f"{algorithm.name}.select_arm() returned {arm}; expected an arm in [0, {n_arms})"
            )

        reward = float(problem.sample(arm, reward_rng))
        if not np.isfinite(reward):
            raise ValueError("problem.sample() must return a finite reward")

        current_expected = np.asarray(problem.expected_rewards, dtype=float)
        if current_expected.shape != (n_arms,):
            raise ValueError(f"problem.expected_rewards must have shape ({n_arms},)")
        if step == 0 and not np.all(np.isfinite(current_expected)):
            raise ValueError("problem.expected_rewards must be finite")
        expected_of_selected[step] = current_expected[arm]
        expected_of_best[step] = current_expected.max()

        algorithm.update(arm, reward)
        rewards[step] = reward
        selected_arms[step] = arm

    estimated_values = np.asarray(algorithm.estimated_values, dtype=float)
    total_pulls = np.asarray(algorithm.total_pulls, dtype=int)
    if estimated_values.shape != (n_arms,):
        raise ValueError(f"{algorithm.name}.estimated_values must have shape ({n_arms},)")
    if total_pulls.shape != (n_arms,):
        raise ValueError(f"{algorithm.name}.total_pulls must have shape ({n_arms},)")
    observed_pulls = np.bincount(selected_arms, minlength=n_arms)
    if not np.array_equal(total_pulls, observed_pulls):
        raise ValueError(f"{algorithm.name}.total_pulls is inconsistent with selected arms")

    return RunResult(
        algorithm_name=algorithm.name,
        problem_name=problem.name,
        rewards=_readonly(rewards),
        selected_arms=_readonly(selected_arms),
        expected_reward_of_selected=_readonly(expected_of_selected),
        expected_reward_of_best=_readonly(expected_of_best),
        estimated_values=_readonly(estimated_values),
        total_pulls=_readonly(total_pulls),
    )
