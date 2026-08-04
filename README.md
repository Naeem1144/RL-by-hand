# RL by hand

Reinforcement learning algorithms implemented from scratch with NumPy and
Matplotlib.

## Project structure

```
algorithms/           # bandit algorithms, each self-contained
  epsilon_greedy.py   #   epsilon-greedy (constant/decaying, optimistic init)
  ucb.py              #   UCB1 (upper confidence bound)
  thompson.py         #   Thompson sampling (Beta-Bernoulli)
bandit_utils.py       # shared metrics + multi-seed averaging
visualizations.py     # dashboard plotter
compare_bandits.py    # epsilon-greedy sweep comparison
compare_algorithms.py # epsilon-greedy vs UCB1 vs Thompson comparison
images/               # canonical figures used in this README
```

## Epsilon-greedy multi-armed bandit

`algorithms/epsilon_greedy.py` runs a Bernoulli multi-armed bandit (1,000 arms,
1,000 steps, ε = 0.10) where the agent explores a random arm with probability ε
and otherwise exploits its current estimate of the best arm. Arm values are
updated incrementally with the running-average rule.

```
python -m algorithms.epsilon_greedy
```

Each run saves a dashboard to `images/bandit_results.png`.

![Epsilon-greedy multi-armed bandit results](images/bandit_results.png)

## Decaying epsilon and optimistic initialization

`run_bandit` accepts a `decay` flag plus a `decay_rate`: when
`decay=True, decay_rate=0.99`, ε is multiplied by the decay factor after every
step so the agent explores less as it learns. The `optimistic_initialization`
flag starts all estimates at 1.0 (the maximum reward) instead of 0.0, which
forces early exploration of every arm.

`compare_bandits.py` compares constant vs decaying epsilon, with and without
optimistic initialization, across a grid of arm and step counts and a range of
decay rates — every configuration averaged over 10 seeds for robust results —
printing a summary table and saving a comparison figure plus CSV:

```
python compare_bandits.py
```

![Constant vs decaying epsilon comparison](images/comparison.png)

## UCB1

`algorithms/ucb.py` implements UCB1 from scratch: it picks the arm maximizing
`mean + c·√(log t / n)`, balancing exploitation with an uncertainty bonus.

```
python -m algorithms.ucb
```

## Thompson sampling

`algorithms/thompson.py` implements Beta-Bernoulli Thompson sampling: each
arm's reward probability is sampled from a Beta posterior and the best sample
is pulled.

```
python -m algorithms.thompson
```

`compare_algorithms.py` benchmarks the tuned epsilon-greedy from above against
UCB1 and Thompson sampling across the same grid, with every algorithm averaged
over 10 seeds:

```
python compare_algorithms.py
```

![Epsilon-greedy vs UCB1 vs Thompson sampling](images/algorithm_comparison.png)
