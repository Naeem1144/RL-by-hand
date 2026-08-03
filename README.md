# RL by hand

Reinforcement learning algorithms implemented from scratch with NumPy and
Matplotlib.

## Epsilon-greedy multi-armed bandit

`epsilon_greedy.py` runs a Bernoulli multi-armed bandit (1,000 arms, 1,000
steps, ε = 0.10) where the agent explores a random arm with probability ε and
otherwise exploits its current estimate of the best arm. Arm values are
updated incrementally with the running-average rule.

```
python epsilon_greedy.py
```

Each run saves a timestamped dashboard (e.g. `bandit_results_20260803_001201.png`)
so repeated runs don't overwrite each other.

![Epsilon-greedy multi-armed bandit results](images/bandit_results.png)

## Decaying epsilon

`run_bandit` accepts a `decay` parameter: when set (e.g. `decay=0.99`), ε is
multiplied by the decay factor after every step so the agent explores less as
it learns. Passing `decay=None` (the default) keeps ε constant for the run.

`compare_bandits.py` compares constant vs decaying epsilon across a grid of
arm and step counts and a range of decay rates, printing a summary table and
saving a comparison figure plus CSV:

```
python compare_bandits.py
```

![Constant vs decaying epsilon comparison](images/comparison.png)

## Coming soon...

More advanced versions of epsilon-greedy:

- **Optimistic initial values** - start estimates high to encourage early exploration
- **UCB (upper confidence bound)** - pick arms by uncertainty-adjusted estimate
