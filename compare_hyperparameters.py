"""Compare exploration hyperparameters across all implemented bandit families.

The experiment fixes one finite-horizon Bernoulli problem size and evaluates
every configuration on the same 100 seeded problem instances. It sweeps the
epsilon-greedy constant epsilon, decay factor, and initialization, the UCB1
confidence scale, and the Thompson sampling Beta prior concentration
(symmetric ``Beta(a, a)`` priors, so prior mean and strength are not
confounded).

Per-seed results are saved alongside the aggregates so paired significance
tests can be recomputed after the fact. Rankings report 95% confidence
intervals and paired comparisons against the best configuration.
"""

import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.epsilon_greedy import run_bandit
from algorithms.thompson import run_thompson
from algorithms.ucb import run_ucb
from bandit_utils import average_over_seeds


N_ARMS = 100
N_STEPS = 2_000
SEEDS = list(range(100))

# Starting epsilon shared by every decay schedule.
DECAY_EPSILON_0 = 0.10
# Two-sided 95% critical value of Student's t with len(SEEDS) - 1 = 99 dof.
T_CRIT_95 = 1.9842

CONSTANT_EPSILON_GRID = [0.01, 0.03, 0.10, 0.30]
# The decay values match compare_bandits.py.
DECAY_GRID = [0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
UCB_C_GRID = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.05,
    0.075,
    0.10,
    0.20,
    0.50,
    float(np.sqrt(2.0)),
]
# Symmetric Beta(a, a) priors keep the prior mean fixed at the correct value
# 0.5 while varying concentration, isolating prior strength from prior mean.
THOMPSON_CONCENTRATION_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

RunFn = Callable[[int], dict[str, np.ndarray]]


def build_run_specs() -> list[tuple[str, str, RunFn, dict[str, object]]]:
    """Return labeled runners and plotting metadata for every configuration."""
    specs: list[tuple[str, str, RunFn, dict[str, object]]] = []

    for optimistic in (False, True):
        initialization = "optimistic" if optimistic else "zero-init"

        for epsilon in CONSTANT_EPSILON_GRID:
            variant = f"{initialization}, constant eps={epsilon:g}"

            def run_constant_epsilon(
                seed: int,
                optimistic: bool = optimistic,
                epsilon: float = epsilon,
            ) -> dict[str, np.ndarray]:
                return run_bandit(
                    n_arms=N_ARMS,
                    n_steps=N_STEPS,
                    epsilon=epsilon,
                    seed=seed,
                    optimistic_initialization=optimistic,
                )

            specs.append(
                (
                    "epsilon-greedy",
                    variant,
                    run_constant_epsilon,
                    {"init": initialization, "schedule": "constant", "x": epsilon},
                )
            )

        for decay in DECAY_GRID:
            variant = f"{initialization}, decay={decay:g}"

            def run_decaying_epsilon(
                seed: int,
                optimistic: bool = optimistic,
                decay: float = decay,
            ) -> dict[str, np.ndarray]:
                return run_bandit(
                    n_arms=N_ARMS,
                    n_steps=N_STEPS,
                    epsilon=DECAY_EPSILON_0,
                    seed=seed,
                    decay=True,
                    decay_rate=decay,
                    optimistic_initialization=optimistic,
                )

            specs.append(
                (
                    "epsilon-greedy",
                    variant,
                    run_decaying_epsilon,
                    {"init": initialization, "schedule": "decay", "x": decay},
                )
            )

    for c in UCB_C_GRID:
        variant = "c=sqrt(2)" if np.isclose(c, np.sqrt(2.0)) else f"c={c:g}"

        def run_confidence_bound(seed: int, c: float = c) -> dict[str, np.ndarray]:
            return run_ucb(
                n_arms=N_ARMS,
                n_steps=N_STEPS,
                c=c,
                seed=seed,
            )

        specs.append(("UCB1", variant, run_confidence_bound, {"x": c}))

    for concentration in THOMPSON_CONCENTRATION_GRID:
        variant = f"Beta({concentration:g},{concentration:g})"

        def run_posterior_sampling(
            seed: int,
            concentration: float = concentration,
        ) -> dict[str, np.ndarray]:
            return run_thompson(
                n_arms=N_ARMS,
                n_steps=N_STEPS,
                seed=seed,
                prior_alpha=concentration,
                prior_beta=concentration,
            )

        specs.append(("Thompson sampling", variant, run_posterior_sampling, {"x": concentration}))

    return specs


def evaluate() -> list[dict[str, object]]:
    """Evaluate every configuration and return aggregate plus per-seed metrics."""
    rows: list[dict[str, object]] = []
    n_seeds = len(SEEDS)

    for family, variant, run_fn, meta in build_run_specs():
        (
            _,
            _,
            _,
            avg_reward,
            std_reward,
            total_regret,
            std_regret,
            per_seed_reward,
            per_seed_regret,
        ) = average_over_seeds(run_fn, SEEDS)
        row: dict[str, object] = {
            "family": family,
            "variant": variant,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "se_reward": std_reward / np.sqrt(n_seeds),
            "total_regret": total_regret,
            "std_regret": std_regret,
            "se_regret": std_regret / np.sqrt(n_seeds),
            "per_seed_reward": per_seed_reward,
            "per_seed_regret": per_seed_regret,
        }
        row.update(meta)
        rows.append(row)

    return rows


def annotate_significance(rows: list[dict[str, object]]) -> dict[str, object]:
    """Add 95% CIs and paired seed-matched comparisons against the best row."""
    best = min(rows, key=lambda row: float(row["total_regret"]))

    for row in rows:
        half_width = T_CRIT_95 * float(row["se_regret"])
        row["ci_regret_low"] = float(row["total_regret"]) - half_width
        row["ci_regret_high"] = float(row["total_regret"]) + half_width

        delta = row["per_seed_regret"] - best["per_seed_regret"]
        mean_delta = float(delta.mean())
        se_delta = float(delta.std(ddof=1) / np.sqrt(delta.size))
        row["delta_vs_best"] = mean_delta
        row["delta_ci_low"] = mean_delta - T_CRIT_95 * se_delta
        row["delta_ci_high"] = mean_delta + T_CRIT_95 * se_delta
        # Significantly worse only if the entire paired CI lies above zero.
        row["sig_worse_than_best"] = bool(row["delta_ci_low"] > 0.0)

    return best


def random_policy_regret() -> float:
    """Expected pseudo-regret of the uniform-random policy on the same instances."""
    regrets: list[float] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        true_probs = rng.random(N_ARMS)
        regrets.append(float(N_STEPS * (true_probs.max() - true_probs.mean())))
    return float(np.mean(regrets))


def print_results(rows: list[dict[str, object]]) -> None:
    """Print configurations ranked by mean cumulative pseudo-regret."""
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))
    header = (
        f"{'rank':>4} {'family':<18} {'variant':<30} {'avg reward':>10} "
        f"{'regret':>8} {'regret 95% CI':>17} {'d(best)':>8} {'d(best) 95% CI':>17} "
        f"{'sig worse':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(ranked, start=1):
        sig = "yes" if row["sig_worse_than_best"] else "-"
        print(
            f"{rank:>4} {str(row['family']):<18} {str(row['variant']):<30} "
            f"{float(row['avg_reward']):>10.4f} "
            f"{float(row['total_regret']):>8.1f} "
            f"[{float(row['ci_regret_low']):>7.1f}, {float(row['ci_regret_high']):>7.1f}] "
            f"{float(row['delta_vs_best']):>8.1f} "
            f"[{float(row['delta_ci_low']):>7.1f}, {float(row['delta_ci_high']):>7.1f}] "
            f"{sig:>9}"
        )


def save_csv(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Save aggregate metrics, CIs, and significance flags in rank order."""
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "rank",
        "family",
        "variant",
        "avg_reward",
        "std_reward",
        "se_reward",
        "total_regret",
        "std_regret",
        "se_regret",
        "ci95_regret_low",
        "ci95_regret_high",
        "delta_vs_best",
        "delta_ci95_low",
        "delta_ci95_high",
        "sig_worse_than_best",
    ]
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "family": row["family"],
                    "variant": row["variant"],
                    "avg_reward": row["avg_reward"],
                    "std_reward": row["std_reward"],
                    "se_reward": row["se_reward"],
                    "total_regret": row["total_regret"],
                    "std_regret": row["std_regret"],
                    "se_regret": row["se_regret"],
                    "ci95_regret_low": row["ci_regret_low"],
                    "ci95_regret_high": row["ci_regret_high"],
                    "delta_vs_best": row["delta_vs_best"],
                    "delta_ci95_low": row["delta_ci_low"],
                    "delta_ci95_high": row["delta_ci_high"],
                    "sig_worse_than_best": row["sig_worse_than_best"],
                }
            )

    return csv_path


def save_per_seed_csv(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Save per-seed outcomes so paired tests can be recomputed later."""
    csv_path = output_dir / "per_seed.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "variant", "seed", "total_regret", "final_reward"])
        for row in rows:
            for seed, regret, reward in zip(
                SEEDS, row["per_seed_regret"], row["per_seed_reward"]
            ):
                writer.writerow([row["family"], row["variant"], seed, regret, reward])

    return csv_path


def plot_results(rows: list[dict[str, object]], random_regret: float) -> Path:
    """Plot regret and reward sensitivity for the three policy families."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.facecolor": "#F8FAFC",
            "figure.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(23, 9))
    fig.subplots_adjust(
        left=0.05,
        right=0.99,
        top=0.84,
        bottom=0.14,
        hspace=0.34,
        wspace=0.28,
    )

    def epsilon_rows(schedule: str, optimistic: bool) -> list[dict[str, object]]:
        initialization = "optimistic" if optimistic else "zero-init"
        return sorted(
            (
                row
                for row in rows
                if row["family"] == "epsilon-greedy"
                and row["schedule"] == schedule
                and row["init"] == initialization
            ),
            key=lambda row: float(row["x"]),
        )

    def plot_series(
        ax,
        x_values,
        selected: list[dict[str, object]],
        metric: str,
        **style,
    ) -> None:
        se_key = "se_regret" if metric == "total_regret" else "se_reward"
        ax.errorbar(
            x_values,
            [float(row[metric]) for row in selected],
            yerr=[T_CRIT_95 * float(row[se_key]) for row in selected],
            capsize=3,
            linewidth=2,
            **style,
        )

    init_styles = ((False, "#64748B", "o"), (True, "#2563EB", "s"))

    eps_values = np.array(CONSTANT_EPSILON_GRID)
    for optimistic, color, marker in init_styles:
        selected = epsilon_rows("constant", optimistic)
        label = "Optimistic init" if optimistic else "Zero init"
        plot_series(
            axes[0, 0], eps_values, selected, "total_regret",
            color=color, marker=marker, label=label,
        )
        plot_series(
            axes[1, 0], eps_values, selected, "avg_reward",
            color=color, marker=marker, label=label,
        )

    x_decay = np.arange(len(DECAY_GRID))
    for optimistic, color, marker in init_styles:
        selected = epsilon_rows("decay", optimistic)
        label = "Optimistic init" if optimistic else "Zero init"
        plot_series(
            axes[0, 1], x_decay, selected, "total_regret",
            color=color, marker=marker, label=label,
        )
        plot_series(
            axes[1, 1], x_decay, selected, "avg_reward",
            color=color, marker=marker, label=label,
        )

    ucb_rows = sorted(
        (row for row in rows if row["family"] == "UCB1"),
        key=lambda row: float(row["x"]),
    )
    c_values = np.array([float(row["x"]) for row in ucb_rows])
    plot_series(axes[0, 2], c_values, ucb_rows, "total_regret", color="#7C3AED", marker="o")
    plot_series(axes[1, 2], c_values, ucb_rows, "avg_reward", color="#7C3AED", marker="o")

    thompson_rows = sorted(
        (row for row in rows if row["family"] == "Thompson sampling"),
        key=lambda row: float(row["x"]),
    )
    a_values = np.array([float(row["x"]) for row in thompson_rows])
    plot_series(axes[0, 3], a_values, thompson_rows, "total_regret", color="#F97316", marker="o")
    plot_series(axes[1, 3], a_values, thompson_rows, "avg_reward", color="#F97316", marker="o")

    for col in range(4):
        axes[0, col].axhline(
            random_regret,
            color="#DC2626",
            linestyle="--",
            linewidth=1.5,
            label="Random policy" if col == 0 else None,
        )

    for row_i, ylabel in (
        (0, "Mean cumulative pseudo-regret"),
        (1, "Mean final average reward"),
    ):
        axes[row_i, 0].set(title="Epsilon-greedy · constant ε", ylabel=ylabel)
        axes[row_i, 1].set(title="Epsilon-greedy · decay (ε₀=0.1)")
        axes[row_i, 2].set(title="UCB1")
        axes[row_i, 3].set(title="Thompson sampling")

        axes[row_i, 0].set_xscale("log")
        axes[row_i, 0].set_xticks(eps_values, [f"{epsilon:g}" for epsilon in eps_values])
        axes[row_i, 1].set_xticks(x_decay, [f"{decay:g}" for decay in DECAY_GRID])
        axes[row_i, 2].set_xscale("log")
        axes[row_i, 2].set_xticks(
            c_values,
            [
                "sqrt(2)" if np.isclose(c, np.sqrt(2.0)) else f"{c:g}"
                for c in c_values
            ],
            rotation=45,
            ha="right",
        )
        axes[row_i, 3].set_xscale("log")
        axes[row_i, 3].set_xticks(a_values, [f"{a:g}" for a in a_values])

        for col in range(4):
            axes[row_i, col].grid(True, which="both")

    for col in (0, 1):
        for row_i in (0, 1):
            axes[row_i, col].legend(frameon=True, facecolor="white", framealpha=0.95)

    axes[1, 0].set_xlabel("Constant epsilon ε")
    axes[1, 1].set_xlabel("Decay factor d")
    axes[1, 2].set_xlabel("Confidence scale c", labelpad=16)
    axes[1, 3].set_xlabel("Prior concentration a (Beta(a, a))")

    best = min(rows, key=lambda row: float(row["total_regret"]))
    fig.suptitle(
        "Bandit hyperparameter ablation",
        y=0.965,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.915,
        f"K={N_ARMS} arms, T={N_STEPS:,} steps, averaged over {len(SEEDS)} matched seeds; "
        "error bars are 95% CIs",
        ha="center",
        fontsize=12,
        color="#64748B",
    )
    fig.text(
        0.5,
        0.035,
        "Best configuration: "
        f"{best['family']} ({best['variant']})  |  "
        f"reward={float(best['avg_reward']):.4f}  |  "
        f"pseudo-regret={float(best['total_regret']):.1f}  |  "
        f"random-policy regret={random_regret:.1f}",
        ha="center",
        fontsize=10,
        color="#64748B",
    )

    figure_path = Path("images/hyperparameter_comparison.png")
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return figure_path


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"hyperparameter_comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate()
    annotate_significance(rows)
    random_regret = random_policy_regret()

    print_results(rows)
    print(f"\nRandom-policy reference regret: {random_regret:.1f}")

    csv_path = save_csv(rows, output_dir)
    per_seed_path = save_per_seed_csv(rows, output_dir)
    figure_path = plot_results(rows, random_regret)

    print(f"\nSaved summary to {csv_path.resolve()}")
    print(f"Saved per-seed results to {per_seed_path.resolve()}")
    print(f"Saved figure to {figure_path.resolve()}")


if __name__ == "__main__":
    main()
