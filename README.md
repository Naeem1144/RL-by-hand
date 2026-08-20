# RL by Hand

Small, interchangeable multi-armed bandit algorithms implemented with NumPy.

The core has three concepts:

```text
Algorithm  <── evaluation runner ──>  Problem
```

- `algorithms/`: one algorithm per file
- `problems/`: reward-generating environments (stationary Bernoulli and
  Gaussian, plus a drifting Bernoulli whose arm means random-walk)
- `evaluation/`: one shared runner, comparison API, and paired statistics
- `study/`: one reproducible multi-scenario comparison of the included
  algorithms

## Install and test

```bash
uv sync --locked
uv run pytest
uv run ruff check .
```

## Run one algorithm

```python
import numpy as np

from algorithms import UCB
from evaluation import run
from problems import BernoulliBandit

problem = BernoulliBandit(np.array([0.1, 0.25, 0.8]))
result = run(UCB(), problem, n_steps=1_000, seed=7)

print(result.average_reward)
print(result.total_regret)
print(result.total_pulls)
```

## Compare algorithms

```python
import numpy as np

from algorithms import UCB, EpsilonGreedy, ThompsonSampling
from evaluation import compare
from problems import BernoulliBandit

problem = BernoulliBandit(np.array([0.1, 0.25, 0.8]))
comparison = compare(
    {
        "epsilon-greedy": EpsilonGreedy(),
        "ucb": UCB(),
        "thompson": ThompsonSampling(),
    },
    problem,
    n_steps=1_000,
    seeds=range(30),
)

for row in comparison.summaries():
    print(row.name, row.mean_reward, row.mean_regret)
```

Every algorithm sees the same problem and matched random seeds in each run.

## Comparison study

The repository contains one focused hyperparameter study instead of several
overlapping benchmark scripts. It evaluates 35 configurations across three
scenarios — a fixed ten-arm Bernoulli problem, a fixed ten-arm Gaussian
problem, and a drifting ten-arm Bernoulli problem — for 2,000 steps over 50
matched runs each:

- Bernoulli (17 configurations): epsilon-greedy with `epsilon` in
  `{0.01, 0.05, 0.1, 0.2, 0.3}`, two decayed-epsilon variants, one optimistic
  initialization, UCB with `c` in `{0.1, 0.5, 1.0, sqrt(2), 2.0}`, and
  Thompson sampling with symmetric Beta priors in `{0.5, 1.0, 2.0, 5.0}`
- Gaussian (10 configurations): epsilon-greedy and UCB subsets, an optimistic
  initialization, and normal Thompson sampling with prior width
  `sigma0` in `{0.5, 1.0, 2.0}`
- Drifting (8 configurations): fixed versus recency-weighted epsilon-greedy,
  plus UCB and Thompson references

Run it with:

```bash
uv run rl-study
```

Best configuration from each family on the fixed Bernoulli problem:

| Rank | Configuration | Mean reward | Mean pseudo-regret |
| ---: | --- | ---: | ---: |
| 1 | UCB `c=0.5` | 0.5163 ± 0.0056 | 61.68 ± 9.07 |
| 2 | Thompson `Beta(0.5, 0.5)` | 0.5134 ± 0.0048 | 67.85 ± 5.59 |
| 6 | Epsilon-greedy `epsilon=0.1` | 0.4864 ± 0.0116 | 121.87 ± 22.34 |

On the Gaussian problem the top two are a statistical tie (paired difference
-1.3 ± 25.9): Thompson-Normal `sigma0=2` at regret 64.4 ± 5.1 edges out UCB
`c=1` at 65.7 ± 25.1, while a too-narrow prior (`sigma0=0.5`) is worst.
On the drifting problem recency weighting wins clearly: epsilon-greedy with
`alpha=0.2` reaches regret 216.5 ± 10.9 and beats fixed `epsilon=0.1`
(431.3 ± 38.7) on 96% of matched seeds.

Values after `±` are 95% confidence-interval half-widths. Because every run
shares its seed, each configuration is also compared with the scenario's best
configuration seed by seed; these paired differences are much more sensitive
than comparing independent intervals. These results apply to these
finite-horizon problems; they are not universal algorithm rankings.

![Fixed Bernoulli comparison](study/results/bernoulli/comparison.png)

The reproducible outputs are in `study/results/`:

- `study.json`: manifest with the git commit, run configuration, and grids
- `<scenario>/summary.csv`: all configurations ranked with confidence intervals
- `<scenario>/per_run.csv`: outcomes for every configuration and seed
- `<scenario>/paired.csv`: per-seed paired differences against the best run
- `<scenario>/comparison.png`: endpoint rankings and best-per-family curves

with `<scenario>` one of `bernoulli`, `gaussian`, or `drifting`.

## Add an algorithm

Create `algorithms/my_algorithm.py` and implement `BanditAlgorithm`:

```python
class MyAlgorithm(BanditAlgorithm):
    @property
    def name(self) -> str: ...

    def reset(self, n_arms, rng) -> None: ...
    def select_arm(self, step: int) -> int: ...
    def update(self, arm: int, reward: float) -> None: ...

    @property
    def estimated_values(self): ...

    @property
    def total_pulls(self): ...
```

Export it from `algorithms/__init__.py`, add an isolated
`tests/algorithms/test_my_algorithm.py`, and include its factory in
`tests/algorithms/test_contract.py`. It can then be passed directly to `run`
or `compare`; no simulation loop or benchmark adapter is needed.

## Layout

```text
.
├── algorithms/
│   ├── base.py
│   ├── epsilon_greedy.py
│   ├── thompson.py
│   ├── thompson_normal.py
│   └── ucb.py
├── problems/
│   ├── base.py
│   ├── bernoulli.py
│   ├── drifting.py
│   └── gaussian.py
├── evaluation/
│   ├── comparison.py
│   └── runner.py
├── study/
│   ├── compare.py
│   └── results/
│       ├── bernoulli/
│       ├── drifting/
│       └── gaussian/
└── tests/
```
