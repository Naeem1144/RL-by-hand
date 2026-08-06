For each seed $s$, I calculated:

$$
\bar r_s=\frac{1}{T}\sum_{t=1}^{T}r_{s,t}
$$

and cumulative pseudo-regret:

$$
R_s=\sum_{t=1}^{T}\left(\mu_s^\star-\mu_{s,a_t}\right).
$$

Then, across $n=100$ seeds, the reported mean is

$$
\bar x=\frac{1}{n}\sum_{s=1}^{n}x_s,
$$

and the reported deviation uses NumPy’s sample standard deviation:

$$
s_x=
\sqrt{\frac{1}{n-1}\sum_{s=1}^{n}(x_s-\bar x)^2}.
$$

These standard deviations describe variation between runs; they are not
confidence intervals for the mean.

A 95% confidence interval for the mean would instead be approximately

$$
\bar x \pm t_{0.975,99}\frac{s}{\sqrt{100}},
$$

where $s$ uses `ddof=1` and $t_{0.975,99}\approx1.984$.

The hyperparameter benchmark selects its reference on a separate tuning set.
On the held-out set, paired differences use simultaneous max-t bootstrap
intervals so the full family of comparisons is covered. These statistical
intervals are separate from the confidence bonus used internally by UCB(c).
