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

## Coming soon...

More advanced versions of epsilon-greedy and related exploration strategies:

- **Decaying epsilon** - anneal ε over time to explore less as the agent learns
- **Optimistic initial values** - start estimates high to encourage early exploration
- **UCB1** - choose arms by upper confidence bound instead of random exploration
- **Thompson sampling** - sample from a posterior distribution over arm values
- **Gradient bandits** - learn arm preferences with softmax action selection
- **Contextual bandits** - pick arms based on observable context features
- **Q-learning & SARSA** - move from bandits to full MDPs with ε-greedy policies
- **DQN** - deep Q-networks with ε-greedy exploration for Atari-style tasks
