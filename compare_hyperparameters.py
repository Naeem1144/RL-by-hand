"""Compare exploration hyperparameters across all implemented bandit families.

The experiment fixes one finite-horizon Bernoulli problem size and evaluates
every configuration on the same 100 seeded problem instances. It sweeps
epsilon decay and initialization, the UCB1 confidence scale, and the Thompson
sampling Beta prior. Results are printed, saved as CSV, and summarized in a
canonical figure for the README.
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
EPSILON = 0.10

# None denotes constant epsilon. The remaining values match compare_bandits.py.
DECAY_GRID: list[float | None] = [
    None,
    0.90,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999,
]
UCB_C_GRID = [0.01, 0.03, 0.05, 0.075, 0.10, 0.20, 0.50, float(np.sqrt(2.0))]
THOMPSON_PRIOR_GRID = [
    (0.5, 0.5),
    (1.0, 1.0),
    (2.0, 2.0),
    (5.0, 5.0),
    (2.0, 1.0),
    (5.0, 1.0),
]

RunFn = Callable[[int], dict[str, np.ndarray]]
ResultRow = dict[str, str | float]


def build_run_specs() -> list[tuple[str, str, RunFn]]:
    """Return labeled runners for every hyperparameter configuration."""
    specs: list[tuple[str, str, RunFn]] = []

    for optimistic in (False, True):
        initialization = "optimistic" if optimistic else "zero-init"
        for decay in DECAY_GRID:
            schedule = "constant" if decay is None else f"decay={decay:g}"
            variant = f"{initialization}, {schedule}"

            def run_epsilon(
                seed: int,
                optimistic: bool = optimistic,
                decay: float | None = decay,
            ) -> dict[str, np.ndarray]:
                return run_bandit(
                    n_arms=N_ARMS,
                    n_steps=N_STEPS,
                    epsilon=EPSILON,
                    seed=seed,
                    decay=decay is not None,
                    decay_rate=decay,
                    optimistic_initialization=optimistic,
                )

            specs.append(("epsilon-greedy", variant, run_epsilon))

    for c in UCB_C_GRID:
        variant = "c=sqrt(2)" if np.isclose(c, np.sqrt(2.0)) else f"c={c:g}"

        def run_confidence_bound(seed: int, c: float = c) -> dict[str, np.ndarray]:
            return run_ucb(
                n_arms=N_ARMS,
                n_steps=N_STEPS,
                c=c,
                seed=seed,
            )

        specs.append(("UCB1", variant, run_confidence_bound))

    for prior_alpha, prior_beta in THOMPSON_PRIOR_GRID:
        variant = f"Beta({prior_alpha:g},{prior_beta:g})"

        def run_posterior_sampling(
            seed: int,
            prior_alpha: float = prior_alpha,
            prior_beta: float = prior_beta,
        ) -> dict[str, np.ndarray]:
            return run_thompson(
                n_arms=N_ARMS,
                n_steps=N_STEPS,
                seed=seed,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
            )

        specs.append(("Thompson sampling", variant, run_posterior_sampling))

    return specs


def evaluate() -> list[ResultRow]:
    """Evaluate every configuration and return aggregate metrics."""
    rows: list[ResultRow] = []

    for family, variant, run_fn in build_run_specs():
        (
            _,
            _,
            _,
            avg_reward,
            std_reward,
            total_regret,
            std_regret,
        ) = average_over_seeds(run_fn, SEEDS)
        rows.append(
            {
                "family": family,
                "variant": variant,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "total_regret": total_regret,
                "std_regret": std_regret,
            }
        )

    return rows


def print_results(rows: list[ResultRow]) -> None:
    """Print configurations ranked by mean cumulative pseudo-regret."""
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))
    header = (
        f"{'rank':>4} {'family':<18} {'variant':<27} "
        f"{'avg reward':>10} {'rew std':>8} {'regret':>10} {'reg std':>8}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank:>4} {str(row['family']):<18} {str(row['variant']):<27} "
            f"{float(row['avg_reward']):>10.4f} "
            f"{float(row['std_reward']):>8.4f} "
            f"{float(row['total_regret']):>10.1f} "
            f"{float(row['std_regret']):>8.1f}"
        )


def save_csv(rows: list[ResultRow], output_dir: Path) -> Path:
    """Save aggregate metrics in rank order."""
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "rank",
        "family",
        "variant",
        "avg_reward",
        "std_reward",
        "total_regret",
        "std_regret",
    ]
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            writer.writerow({"rank": rank, **row})

    return csv_path


def plot_results(rows: list[ResultRow]) -> Path:
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

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.84,
        bottom=0.12,
        hspace=0.34,
        wspace=0.25,
    )

    epsilon_rows = [row for row in rows if row["family"] == "epsilon-greedy"]
    ucb_rows = [row for row in rows if row["family"] == "UCB1"]
    thompson_rows = [row for row in rows if row["family"] == "Thompson sampling"]

    schedule_labels = [
        "constant" if decay is None else f"{decay:g}" for decay in DECAY_GRID
    ]
    x_epsilon = np.arange(len(schedule_labels))
    for optimistic, color, marker in (
        (False, "#64748B", "o"),
        (True, "#2563EB", "s"),
    ):
        initialization = "optimistic" if optimistic else "zero-init"
        selected = [
            next(
                row
                for row in epsilon_rows
                if str(row["variant"]).startswith(initialization)
                and (
                    (decay is None and str(row["variant"]).endswith("constant"))
                    or (
                        decay is not None
                        and str(row["variant"]).endswith(f"decay={decay:g}")
                    )
                )
            )
            for decay in DECAY_GRID
        ]
        label = "Optimistic init" if optimistic else "Zero init"
        axes[0, 0].plot(
            x_epsilon,
            [float(row["total_regret"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )
        axes[1, 0].plot(
            x_epsilon,
            [float(row["avg_reward"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )

    c_values = np.array(UCB_C_GRID)
    axes[0, 1].plot(
        c_values,
        [float(row["total_regret"]) for row in ucb_rows],
        color="#7C3AED",
        marker="o",
        linewidth=2,
    )
    axes[1, 1].plot(
        c_values,
        [float(row["avg_reward"]) for row in ucb_rows],
        color="#7C3AED",
        marker="o",
        linewidth=2,
    )

    prior_labels = [f"({alpha:g},{beta:g})" for alpha, beta in THOMPSON_PRIOR_GRID]
    x_thompson = np.arange(len(prior_labels))
    axes[0, 2].plot(
        x_thompson,
        [float(row["total_regret"]) for row in thompson_rows],
        color="#F97316",
        marker="o",
        linewidth=2,
    )
    axes[1, 2].plot(
        x_thompson,
        [float(row["avg_reward"]) for row in thompson_rows],
        color="#F97316",
        marker="o",
        linewidth=2,
    )

    for row, ylabel in (
        (0, "Mean cumulative pseudo-regret"),
        (1, "Mean final average reward"),
    ):
        axes[row, 0].set(
            title="Epsilon-greedy",
            ylabel=ylabel,
            xticks=x_epsilon,
            xticklabels=schedule_labels,
        )
        axes[row, 0].legend(frameon=True, facecolor="white", framealpha=0.95)
        axes[row, 1].set(title="UCB1", xscale="log")
        ucb_labels = [
            "sqrt(2)" if np.isclose(c, np.sqrt(2.0)) else f"{c:g}"
            for c in c_values
        ]
        axes[row, 1].set_xticks(c_values, ucb_labels)
        axes[row, 2].set(
            title="Thompson sampling",
            xticks=x_thompson,
            xticklabels=prior_labels,
        )
        for col in range(3):
            axes[row, col].grid(True, which="both")

    axes[1, 0].set_xlabel("Decay factor d")
    axes[1, 1].set_xlabel("Confidence scale c")
    axes[1, 2].set_xlabel("Beta prior (alpha, beta)")

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
        f"K={N_ARMS} arms, T={N_STEPS:,} steps, averaged over {len(SEEDS)} matched seeds",
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
        f"pseudo-regret={float(best['total_regret']):.1f}",
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
    print_results(rows)
    csv_path = save_csv(rows, output_dir)
    figure_path = plot_results(rows)

    print(f"\nSaved summary to {csv_path.resolve()}")
    print(f"Saved figure to {figure_path.resolve()}")


if __name__ == "__main__":
    main()
