"""Compare exploration hyperparameters across all implemented bandit families.

The experiment fixes one finite-horizon Bernoulli problem size, tunes on 100
problem instances, and confirms the selected reference on 100 disjoint held-out
instances. It sweeps the
epsilon-greedy constant epsilon, decay factor, and initialization, the UCB(c)
confidence scale, and the Thompson sampling Beta prior concentration
(symmetric ``Beta(a, a)`` priors, so prior mean and strength are not
confounded).

Per-instance results are saved alongside the aggregates. Comparisons against
the tuning-selected reference use seed-matched simultaneous bootstrap
intervals, controlling the family-wise error rate across the full grid.
"""

import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.epsilon_greedy import run_bandit
from algorithms.thompson import run_thompson
from algorithms.ucb import run_ucb
from bandit_utils import (
    average_over_seeds,
    generate_problem,
    simulation_seeds,
    write_manifest,
)

N_ARMS = 100
N_STEPS = 2_000
TUNING_ENVIRONMENT_SEEDS = list(range(100))
EVALUATION_ENVIRONMENT_SEEDS = list(range(10_000, 10_100))

# Starting epsilon shared by every decay schedule.
DECAY_EPSILON_0 = 0.10
# Two-sided 95% critical value of Student's t with 99 degrees of freedom.
T_CRIT_95 = 1.9842
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_806

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
# Symmetric Beta(a, a) priors keep their mean fixed at 0.5 while varying
# concentration, isolating prior strength from prior mean.
THOMPSON_CONCENTRATION_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

RunFn = Callable[[int], dict[str, np.ndarray]]


def _matched_inputs(environment_seed: int) -> tuple[np.ndarray, int, int]:
    """Return one explicit problem plus independent policy and reward seeds."""
    true_probs = generate_problem(N_ARMS, environment_seed)
    policy_seed, reward_seed = simulation_seeds(environment_seed)
    return true_probs, policy_seed, reward_seed


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
                true_probs, policy_seed, reward_seed = _matched_inputs(seed)
                return run_bandit(
                    n_arms=N_ARMS,
                    n_steps=N_STEPS,
                    epsilon=epsilon,
                    true_probs=true_probs,
                    policy_seed=policy_seed,
                    reward_seed=reward_seed,
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
                true_probs, policy_seed, reward_seed = _matched_inputs(seed)
                return run_bandit(
                    n_arms=N_ARMS,
                    n_steps=N_STEPS,
                    epsilon=DECAY_EPSILON_0,
                    true_probs=true_probs,
                    policy_seed=policy_seed,
                    reward_seed=reward_seed,
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
            true_probs, policy_seed, reward_seed = _matched_inputs(seed)
            return run_ucb(
                n_arms=N_ARMS,
                n_steps=N_STEPS,
                c=c,
                true_probs=true_probs,
                policy_seed=policy_seed,
                reward_seed=reward_seed,
            )

        specs.append(("UCB(c)", variant, run_confidence_bound, {"x": c}))

    for concentration in THOMPSON_CONCENTRATION_GRID:
        variant = f"Beta({concentration:g},{concentration:g})"

        def run_posterior_sampling(
            seed: int,
            concentration: float = concentration,
        ) -> dict[str, np.ndarray]:
            true_probs, policy_seed, reward_seed = _matched_inputs(seed)
            return run_thompson(
                n_arms=N_ARMS,
                n_steps=N_STEPS,
                true_probs=true_probs,
                policy_seed=policy_seed,
                reward_seed=reward_seed,
                prior_alpha=concentration,
                prior_beta=concentration,
            )

        specs.append(("Thompson sampling", variant, run_posterior_sampling, {"x": concentration}))

    return specs


def evaluate(environment_seeds: list[int]) -> list[dict[str, object]]:
    """Evaluate every configuration and return aggregate plus per-seed metrics."""
    rows: list[dict[str, object]] = []
    n_seeds = len(environment_seeds)

    for family, variant, run_fn, meta in build_run_specs():
        (
            _,
            _,
            _,
            _,
            _,
            avg_reward,
            std_reward,
            total_regret,
            std_regret,
            per_seed_reward,
            per_seed_regret,
        ) = average_over_seeds(run_fn, environment_seeds)
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


def row_key(row: dict[str, object]) -> tuple[str, str]:
    """Return the stable identity of a configuration row."""
    return str(row["family"]), str(row["variant"])


def annotate_significance(
    rows: list[dict[str, object]],
    reference_key: tuple[str, str],
) -> tuple[dict[str, object], float]:
    """Add marginal CIs and simultaneous comparisons to a fixed reference.

    The reference must have been selected without using ``rows``. A paired,
    studentized max-t bootstrap supplies one critical value for the entire
    family of comparisons, controlling family-wise error at approximately 5%.
    """
    reference = next((row for row in rows if row_key(row) == reference_key), None)
    if reference is None:
        raise ValueError(f"reference configuration not found: {reference_key}")

    n_instances = np.asarray(reference["per_seed_regret"]).size
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(0, n_instances, size=(BOOTSTRAP_REPLICATES, n_instances))
    max_statistics = np.zeros(BOOTSTRAP_REPLICATES)

    for row in rows:
        half_width = T_CRIT_95 * float(row["se_regret"])
        row["ci_regret_low"] = float(row["total_regret"]) - half_width
        row["ci_regret_high"] = float(row["total_regret"]) + half_width

        delta = np.asarray(row["per_seed_regret"]) - np.asarray(reference["per_seed_regret"])
        mean_delta = float(delta.mean())
        se_delta = float(delta.std(ddof=1) / np.sqrt(delta.size))
        row["delta_vs_reference"] = mean_delta
        row["se_delta"] = se_delta

        if se_delta > 0.0:
            centered = delta - mean_delta
            bootstrap_means = centered[resamples].mean(axis=1)
            max_statistics = np.maximum(max_statistics, np.abs(bootstrap_means / se_delta))

    simultaneous_critical = float(np.quantile(max_statistics, 0.95))
    for row in rows:
        mean_delta = float(row["delta_vs_reference"])
        half_width = simultaneous_critical * float(row["se_delta"])
        row["delta_ci_low"] = mean_delta - half_width
        row["delta_ci_high"] = mean_delta + half_width
        row["sig_worse_than_reference"] = bool(row["delta_ci_low"] > 0.0)

    return reference, simultaneous_critical


def random_policy_regret(environment_seeds: list[int]) -> float:
    """Expected pseudo-regret of the uniform-random policy on the same instances."""
    regrets: list[float] = []
    for seed in environment_seeds:
        true_probs = generate_problem(N_ARMS, seed)
        regrets.append(float(N_STEPS * (true_probs.max() - true_probs.mean())))
    return float(np.mean(regrets))


def print_results(
    rows: list[dict[str, object]],
    reference: dict[str, object],
    simultaneous_critical: float,
) -> None:
    """Print configurations ranked by mean cumulative pseudo-regret."""
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))
    header = (
        f"{'rank':>4} {'family':<18} {'variant':<30} {'avg reward':>10} "
        f"{'regret':>8} {'regret 95% CI':>17} {'d(ref)':>8} {'simul 95% CI':>17} "
        f"{'sig worse':>9}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(ranked, start=1):
        sig = "yes" if row["sig_worse_than_reference"] else "-"
        print(
            f"{rank:>4} {str(row['family']):<18} {str(row['variant']):<30} "
            f"{float(row['avg_reward']):>10.4f} "
            f"{float(row['total_regret']):>8.1f} "
            f"[{float(row['ci_regret_low']):>7.1f}, {float(row['ci_regret_high']):>7.1f}] "
            f"{float(row['delta_vs_reference']):>8.1f} "
            f"[{float(row['delta_ci_low']):>7.1f}, {float(row['delta_ci_high']):>7.1f}] "
            f"{sig:>9}"
        )
    print(f"\nReference selected on tuning set: {reference['family']} / {reference['variant']}")
    print(f"Simultaneous bootstrap critical value: {simultaneous_critical:.3f}")


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
        "delta_vs_reference",
        "delta_ci95_low",
        "delta_ci95_high",
        "sig_worse_than_reference",
    ]
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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
                    "delta_vs_reference": row["delta_vs_reference"],
                    "delta_ci95_low": row["delta_ci_low"],
                    "delta_ci95_high": row["delta_ci_high"],
                    "sig_worse_than_reference": row["sig_worse_than_reference"],
                }
            )

    return csv_path


def save_per_seed_csv(
    rows: list[dict[str, object]],
    environment_seeds: list[int],
    output_dir: Path,
    filename: str,
) -> Path:
    """Save per-seed outcomes so paired tests can be recomputed later."""
    csv_path = output_dir / filename

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["family", "variant", "environment_seed", "total_regret", "final_reward"])
        for row in rows:
            for seed, regret, reward in zip(
                environment_seeds,
                row["per_seed_regret"],
                row["per_seed_reward"],
                strict=True,
            ):
                writer.writerow([row["family"], row["variant"], seed, regret, reward])

    return csv_path


def save_tuning_csv(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Save tuning-set aggregates without attaching inferential claims."""
    path = output_dir / "tuning_summary.csv"
    ranked = sorted(rows, key=lambda row: float(row["total_regret"]))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "tuning_rank",
                "family",
                "variant",
                "avg_reward",
                "std_reward",
                "total_regret",
                "std_regret",
            ]
        )
        for rank, row in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    row["family"],
                    row["variant"],
                    row["avg_reward"],
                    row["std_reward"],
                    row["total_regret"],
                    row["std_regret"],
                ]
            )
    return path


def plot_results(
    rows: list[dict[str, object]],
    random_regret: float,
    reference: dict[str, object],
) -> Path:
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
        bottom=0.18,
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
            axes[0, 0],
            eps_values,
            selected,
            "total_regret",
            color=color,
            marker=marker,
            label=label,
        )
        plot_series(
            axes[1, 0],
            eps_values,
            selected,
            "avg_reward",
            color=color,
            marker=marker,
            label=label,
        )

    x_decay = np.arange(len(DECAY_GRID))
    for optimistic, color, marker in init_styles:
        selected = epsilon_rows("decay", optimistic)
        label = "Optimistic init" if optimistic else "Zero init"
        plot_series(
            axes[0, 1],
            x_decay,
            selected,
            "total_regret",
            color=color,
            marker=marker,
            label=label,
        )
        plot_series(
            axes[1, 1],
            x_decay,
            selected,
            "avg_reward",
            color=color,
            marker=marker,
            label=label,
        )

    ucb_rows = sorted(
        (row for row in rows if row["family"] == "UCB(c)"),
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
        axes[row_i, 2].set(title="UCB(c)")
        axes[row_i, 3].set(title="Thompson sampling")

        axes[row_i, 0].set_xscale("log")
        axes[row_i, 0].set_xticks(eps_values, [f"{epsilon:g}" for epsilon in eps_values])
        axes[row_i, 1].set_xticks(x_decay, [f"{decay:g}" for decay in DECAY_GRID])
        axes[row_i, 2].set_xscale("log")
        axes[row_i, 2].set_xticks(
            c_values,
            ["sqrt(2)" if np.isclose(c, np.sqrt(2.0)) else f"{c:g}" for c in c_values],
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

    evaluation_best = min(rows, key=lambda row: float(row["total_regret"]))
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
        f"K={N_ARMS}, T={N_STEPS:,}, arm means ~ Uniform(0,1), "
        f"{len(EVALUATION_ENVIRONMENT_SEEDS)} held-out instances; error bars are marginal 95% CIs",
        ha="center",
        fontsize=12,
        color="#64748B",
    )
    fig.text(
        0.5,
        0.035,
        "Tuning-selected reference: "
        f"{reference['family']} ({reference['variant']}); "
        f"held-out regret={float(reference['total_regret']):.1f}  |  "
        "held-out rank-1: "
        f"{evaluation_best['family']} ({evaluation_best['variant']})  |  "
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
    output_dir = Path("results/hyperparameter_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    tuning_rows = evaluate(TUNING_ENVIRONMENT_SEEDS)
    tuning_selected = min(tuning_rows, key=lambda row: float(row["total_regret"]))
    rows = evaluate(EVALUATION_ENVIRONMENT_SEEDS)
    reference, simultaneous_critical = annotate_significance(rows, row_key(tuning_selected))
    random_regret = random_policy_regret(EVALUATION_ENVIRONMENT_SEEDS)

    print_results(rows, reference, simultaneous_critical)
    print(f"\nRandom-policy reference regret: {random_regret:.1f}")

    csv_path = save_csv(rows, output_dir)
    tuning_path = save_tuning_csv(tuning_rows, output_dir)
    tuning_per_seed_path = save_per_seed_csv(
        tuning_rows, TUNING_ENVIRONMENT_SEEDS, output_dir, "tuning_per_seed.csv"
    )
    per_seed_path = save_per_seed_csv(
        rows, EVALUATION_ENVIRONMENT_SEEDS, output_dir, "evaluation_per_seed.csv"
    )
    figure_path = plot_results(rows, random_regret, reference)
    write_manifest(
        output_dir,
        "hyperparameter tuning and held-out confirmation",
        {
            "n_arms": N_ARMS,
            "n_steps": N_STEPS,
            "tuning_environment_seeds": TUNING_ENVIRONMENT_SEEDS,
            "evaluation_environment_seeds": EVALUATION_ENVIRONMENT_SEEDS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "tuning_selected_reference": list(row_key(tuning_selected)),
            "simultaneous_bootstrap_critical": simultaneous_critical,
        },
    )

    print(f"\nSaved summary to {csv_path.resolve()}")
    print(f"Saved tuning summary to {tuning_path.resolve()}")
    print(f"Saved tuning per-seed results to {tuning_per_seed_path.resolve()}")
    print(f"Saved per-seed results to {per_seed_path.resolve()}")
    print(f"Saved figure to {figure_path.resolve()}")


if __name__ == "__main__":
    main()
