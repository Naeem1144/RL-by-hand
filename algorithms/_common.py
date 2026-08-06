"""Validation and random-stream helpers shared by the bandit implementations."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SimulationInputs:
    """Validated environment and independent policy/reward random streams."""

    true_probs: np.ndarray
    policy_rng: np.random.Generator
    reward_rng: np.random.Generator


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


def prepare_simulation(
    n_arms: int,
    n_steps: int,
    seed: int,
    true_probs: np.ndarray | None,
    policy_seed: int | None,
    reward_seed: int | None,
) -> SimulationInputs:
    """Validate inputs and construct independent environment/random streams.

    ``seed`` is a convenient master seed for standalone runs. Benchmarks can
    provide ``true_probs``, ``policy_seed``, and ``reward_seed`` explicitly so
    problem instances and both sources of simulation randomness are matched
    without relying on the algorithms consuming random numbers in lockstep.
    """
    n_arms = _positive_integer(n_arms, "n_arms")
    _positive_integer(n_steps, "n_steps")
    master_seed = _seed_sequence(seed, "seed")
    environment_sequence, policy_sequence, reward_sequence = master_seed.spawn(3)

    if true_probs is None:
        probabilities = np.random.default_rng(environment_sequence).random(n_arms)
    else:
        probabilities = np.asarray(true_probs, dtype=float)
        if probabilities.shape != (n_arms,):
            raise ValueError(f"true_probs must have shape ({n_arms},)")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("true_probs must contain only finite values")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("true_probs must lie in [0, 1]")
        probabilities = probabilities.copy()

    if policy_seed is not None:
        policy_sequence = _seed_sequence(policy_seed, "policy_seed")
    if reward_seed is not None:
        reward_sequence = _seed_sequence(reward_seed, "reward_seed")

    return SimulationInputs(
        true_probs=probabilities,
        policy_rng=np.random.default_rng(policy_sequence),
        reward_rng=np.random.default_rng(reward_sequence),
    )


def validate_probability(value: float, name: str) -> float:
    """Return a finite probability after validating it lies in [0, 1]."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def validate_nonnegative(value: float, name: str) -> float:
    """Return a finite non-negative scalar."""
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value
