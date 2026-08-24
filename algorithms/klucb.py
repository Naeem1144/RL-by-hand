"""KL upper confidence bound (KL-UCB) bandit algorithm."""

from dataclasses import dataclass, field

import numpy as np

from algorithms.base import BanditAlgorithm


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Bernoulli KL divergence ``p log(p/q) + (1-p) log((1-p)/(1-q))``.

    Uses the convention ``0 log 0 = 0``, so ``kl(p||q)`` is ``inf`` whenever
    ``q`` is on the boundary but ``p`` is not, and ``0`` when ``p == q``.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(p > 0.0, np.where(q > 0.0, p * np.log(p / q), np.inf), 0.0)
        second = np.where(
            p < 1.0, np.where(q < 1.0, (1.0 - p) * np.log((1.0 - p) / (1.0 - q)), np.inf), 0.0
        )
    return first + second


def _kl_ucb_index(mean: np.ndarray, pulls: np.ndarray, budget: float) -> np.ndarray:
    """Largest ``theta`` with ``pulls * kl(mean || theta) <= budget``.

    The maximum lies in ``[mean, 1]``.  Bisection on the (vectorised) KL
    constraint is exact to about 60 bits after fixing the ``0 log 0 - 0``
    conventions in :func:`_kl_divergence`.
    """
    mean = np.asarray(mean, dtype=float)
    pulls = np.asarray(pulls, dtype=float)
    lo = mean.copy()
    hi = np.ones_like(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        feasible = _kl_divergence(mean, mid) * pulls <= budget
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


@dataclass
class KLUCB(BanditAlgorithm):
    """KL-UCB (Cappe et al., 2013) for Bernoulli rewards.

    Replaces UCB1's Hoeffding bonus with the exact KL divergence between the
    empirical mean and the candidate arm mean, which yields a tighter upper
    confidence bound for rare events.  Requires binary rewards, exactly like
    :class:`algorithms.thompson.ThompsonSampling`.
    """

    c: float = 3.0
    _values: np.ndarray = field(init=False, repr=False)
    _pulls: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.c = float(self.c)
        if not np.isfinite(self.c) or self.c < 0.0:
            raise ValueError("c must be finite and non-negative")

    @property
    def name(self) -> str:
        return f"KL-UCB(c={self.c:g})"

    def reset(self, n_arms: int, rng: np.random.Generator) -> None:
        del rng
        self._values = np.zeros(n_arms)
        self._pulls = np.zeros(n_arms, dtype=int)

    def select_arm(self, step: int) -> int:
        untried = np.flatnonzero(self._pulls == 0)
        if untried.size:
            return int(untried[0])
        t = int(step) + 1
        log_t = np.log(t)
        budget = log_t + self.c * np.log(max(1.0, log_t))
        scores = _kl_ucb_index(self._values, self._pulls, budget)
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        if reward not in (0.0, 1.0):
            raise ValueError("KLUCB requires binary rewards")
        self._pulls[arm] += 1
        self._values[arm] += (reward - self._values[arm]) / self._pulls[arm]

    @property
    def estimated_values(self) -> np.ndarray:
        return self._values

    @property
    def total_pulls(self) -> np.ndarray:
        return self._pulls
