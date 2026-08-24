"""Sanity checks that UCB regret grows logarithmically, not linearly."""

import numpy as np

from algorithms import UCB
from evaluation import run
from problems import BernoulliBandit

N_STEPS = 5_000
SEEDS = tuple(range(5))


def _regret_rates() -> tuple[float, float]:
    problem = BernoulliBandit(np.array([0.1, 0.9]))
    first_quarter, last_quarter = [], []
    for seed in SEEDS:
        result = run(UCB(), problem, n_steps=N_STEPS, seed=seed)
        cumulative = result.cumulative_regret
        quarter = N_STEPS // 4
        steps = np.arange(1, N_STEPS + 1)
        first_quarter.append(cumulative[quarter - 1] / steps[quarter - 1])
        last_quarter.append((cumulative[-1] - cumulative[quarter]) / (steps[-1] - steps[quarter]))
    return float(np.mean(first_quarter)), float(np.mean(last_quarter))


def test_ucb_per_step_regret_declines_over_time() -> None:
    early_rate, late_rate = _regret_rates()

    assert late_rate < early_rate / 2.0


def test_ucb_total_regret_is_sublinear() -> None:
    problem = BernoulliBandit(np.array([0.1, 0.9]))
    totals = [run(UCB(), problem, n_steps=N_STEPS, seed=seed).total_regret for seed in SEEDS]

    assert np.mean(totals) < 0.02 * N_STEPS
