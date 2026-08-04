# RL by hand

Reinforcement learning algorithms implemented from scratch with **NumPy** and
**Matplotlib** — no RL libraries, just the math, written by hand.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.8-11557c?logo=python&logoColor=white">
</p>

## Contents

1. [Project structure](#project-structure)
2. [Epsilon-greedy](#epsilon-greedy)
3. [Decaying epsilon & optimistic initialization](#decaying-epsilon--optimistic-initialization)
4. [UCB1](#ucb1)
5. [Thompson sampling](#thompson-sampling)
6. [Regret](#regret)
7. [Benchmarks](#benchmarks)

## Project structure

```
algorithms/            # bandit algorithms, each self-contained
  epsilon_greedy.py    #   epsilon-greedy (constant/decaying, optimistic init)
  ucb.py               #   UCB1 (upper confidence bound)
  thompson.py          #   Thompson sampling (Beta-Bernoulli)
bandit_utils.py        # shared metrics + multi-seed averaging
visualizations.py      # dashboard plotter
compare_bandits.py     # epsilon-greedy sweep comparison
compare_algorithms.py  # epsilon-greedy vs UCB1 vs Thompson comparison
images/                # canonical figures used in this README
```

All algorithms solve the same problem: a **$K$-arm Bernoulli bandit** played
for $T$ steps. Arm $i$ has an unknown mean $\mu_i \in [0, 1]$; at step $t$ the
agent pulls arm $a_t$, observes a reward
$r_t \sim \text{Bernoulli}(\mu_{a_t})$, and updates its estimates.

---

## Epsilon-greedy

### Mathematics

With probability $\varepsilon$ the agent explores a uniformly random arm; with
probability $1 - \varepsilon$ it exploits its current best estimate:

$$
a_t = \begin{cases}
  \text{random arm}      & \text{with probability } \varepsilon \\
  \arg\max_{i}\, \hat{Q}_t(i) & \text{with probability } 1 - \varepsilon
\end{cases}
$$

After each pull, the value estimate of the chosen arm is updated with an
incremental running average — a step of constant size $1 / N_t(a_t)$:

$$
\hat{Q}_{t+1}(a_t) = \hat{Q}_t(a_t)
  + \frac{1}{N_t(a_t)} \left( r_t - \hat{Q}_t(a_t) \right)
$$

### Implementation

`algorithms/epsilon_greedy.py` runs a Bernoulli multi-armed bandit with 1,000
arms, 1,000 steps and $\varepsilon = 0.10$.

```bash
python -m algorithms.epsilon_greedy
```

Each run saves a dashboard to `images/bandit_results.png`.

<p align="center">
  <img src="images/bandit_results.png" alt="Epsilon-greedy multi-armed bandit results" width="720">
</p>

---

## Decaying epsilon & optimistic initialization

### Mathematics

Instead of a constant $\varepsilon$, the exploration rate can decay over time so
the agent explores less as it learns:

$$
\varepsilon_t = \varepsilon_0 \cdot d^{\,t}, \qquad d \in (0, 1)
$$

Alternatively, *optimistic initialization* starts every estimate at the maximum
reward, $\hat{Q}_0(i) = 1$ for all arms $i$. Since the running-average update
only ever moves estimates downward, a greedy agent is forced to try every arm
early on.

### Implementation

`run_bandit` in `algorithms/epsilon_greedy.py` accepts a `decay` flag plus a
`decay_rate` (e.g. `decay=True, decay_rate=0.99`), and an
`optimistic_initialization` flag that starts all estimates at 1.0 instead of
0.0.

`compare_bandits.py` compares constant vs decaying epsilon, with and without
optimistic initialization, across a grid of arm and step counts and a range of
decay rates — every configuration averaged over 10 seeds — printing a summary
table and saving a comparison figure plus CSV:

```bash
python compare_bandits.py
```

<p align="center">
  <img src="images/comparison.png" alt="Constant vs decaying epsilon comparison" width="720">
</p>

---

## UCB1

### Mathematics

UCB1 (Auer, Cesa-Bianchi & Fischer, 2002) picks the arm maximizing a value
estimate plus an **exploration bonus** that shrinks as the arm is pulled more
often:

$$
a_t = \arg\max_{i} \left[ \hat{Q}_t(i) + c \sqrt{\frac{\ln t}{N_t(i)}} \right]
$$

The constant $c$ controls how optimistic the agent is (default $c = \sqrt{2}$,
which matches the theoretical bound).

### Implementation

`algorithms/ucb.py` implements UCB1 from scratch. Every arm is pulled once
first so its sample mean is defined, then the arm with the highest
upper-confidence index is chosen on each step.

```bash
python -m algorithms.ucb
```

---

## Thompson sampling

### Mathematics

Thompson sampling is a Bayesian approach: each arm maintains a Beta posterior
$\text{Beta}(\alpha_i, \beta_i)$ over its mean, initialized to a uniform prior
$\alpha_i = \beta_i = 1$. At every step:

1. **Sample** a mean $\tilde{\mu}_i \sim \text{Beta}(\alpha_i, \beta_i)$ for
   every arm.
2. **Pull** the arm with the highest sample, $a_t = \arg\max_i \tilde{\mu}_i$.
3. **Update** the posterior with the observed reward $r_t$:
   $$
   \alpha_{a_t} \leftarrow \alpha_{a_t} + r_t, \qquad
   \beta_{a_t} \leftarrow \beta_{a_t} + (1 - r_t)
   $$

### Implementation

`algorithms/thompson.py` implements Beta-Bernoulli Thompson sampling: each
arm's reward probability is sampled from its Beta posterior and the best sample
is pulled.

```bash
python -m algorithms.thompson
```

---

## Regret

### Mathematics

Every policy is judged by its **regret** — the reward lost by not always
pulling the true best arm:

$$
R_T = T\, \mu^* - \sum_{t=1}^{T} \mu_{a_t}, \qquad \mu^* = \max_i \mu_i
$$

Lower regret means the agent found the best arm faster.

---

## Benchmarks

`compare_algorithms.py` benchmarks the tuned epsilon-greedy (with decaying ε
and optimistic initialization) against UCB1 and Thompson sampling across the
same grid, with every algorithm averaged over 10 seeds:

```bash
python compare_algorithms.py
```

<p align="center">
  <img src="images/algorithm_comparison.png" alt="Epsilon-greedy vs UCB1 vs Thompson sampling" width="720">
</p>
