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

## Results

![Bandit results](bandit_results.png)
