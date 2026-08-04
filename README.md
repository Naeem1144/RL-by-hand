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

## Mathematics

**Setup.** $K$-arm Bernoulli bandit over $T$ steps. Arm $i$ has unknown mean
$\mu_i \in [0,1]$; at step $t$ the agent pulls $a_t$, observes reward
$r_t \sim \text{Bernoulli}(\mu_{a_t})$, and updates its estimates.

### Epsilon-greedy

$$
a_t = \begin{cases}
\text{random arm} & \text{with probability } \varepsilon \\[4pt]
\displaystyle\operatorname*{argmax}_{i}\,\hat{Q}_t(i) & \text{with probability } 1-\varepsilon
\end{cases}
$$

$$
\hat{Q}_{t+1}(a_t) = \hat{Q}_t(a_t) + \frac{1}{N_t(a_t)}\bigl(r_t - \hat{Q}_t(a_t)\bigr)
$$

**Extensions.** Decaying $\varepsilon_t = \varepsilon_0 \cdot d^{\,t}$ reduces
exploration over time. Optimistic initialisation sets $\hat{Q}_0(i)=1$ to force
early exploration.

### UCB1 (Upper Confidence Bound)

$$
a_t = \operatorname*{argmax}_{i}\;\left[\,\hat{Q}_t(i) + c\,\sqrt{\frac{\ln t}{N_t(i)}}\,\right]
$$

The second term is an _exploration bonus_ that shrinks as an arm is pulled more
often. The constant $c$ controls the optimism level (default $c=\sqrt{2}$).

### Thompson sampling (Beta-Bernoulli)

Each arm maintains a Beta posterior $\text{Beta}(\alpha_i,\beta_i)$ (initialised
to $\alpha_i=\beta_i=1$, i.e. uniform prior).

1. **Sample** $\tilde{\mu}_i \sim \text{Beta}(\alpha_i,\beta_i)$ for every arm.
2. **Pull** $a_t = \operatorname*{argmax}_i \tilde{\mu}_i$.
3. **Update** with observed reward $r_t$:
   $$
   \alpha_{a_t} \leftarrow \alpha_{a_t} + r_t,\qquad
   \beta_{a_t} \leftarrow \beta_{a_t} + (1-r_t)
   $$

### Regret

$$
R_T = T\,\mu^* - \sum_{t=1}^{T} \mu_{a_t},\qquad
\mu^* = \max_i \mu_i
$$

Regret measures how much reward was lost by not always pulling the true best arm.

---

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
