"""Smoke test for the scenario-based comparison study."""

from study.compare import SCENARIOS, family_of, run_study, write_results


def test_study_runs_and_writes_expected_bundle(tmp_path) -> None:
    seeds = (0, 1)
    results = run_study(n_steps=20, seeds=seeds)
    paths = write_results(results, seeds=seeds, n_steps=20, output_dir=tmp_path)

    assert set(results) == {scenario.key for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        comparison = results[scenario.key]
        assert set(comparison.runs) == set(scenario.configurations())
        assert {family_of(name) for name in comparison.runs} == {
            "Epsilon-greedy",
            "UCB",
            "Thompson sampling",
        }
        assert all(len(runs) == len(seeds) for runs in comparison.runs.values())

    assert all(path.is_file() for path in paths)
    assert {path.name for path in paths} == {
        "study.json",
        "summary.csv",
        "per_run.csv",
        "paired.csv",
        "comparison.png",
    }


def test_summary_reports_regret_risk_quantiles(tmp_path) -> None:
    results = run_study(n_steps=20, seeds=(0, 1))
    write_results(results, seeds=(0, 1), n_steps=20, output_dir=tmp_path)

    header = (tmp_path / "bernoulli" / "summary.csv").read_text().splitlines()[0]
    for column in ("median_regret", "p90_regret", "max_regret"):
        assert column in header


def test_bernoulli_scenario_sweeps_the_full_grid() -> None:
    bernoulli = next(s for s in SCENARIOS if s.key == "bernoulli")
    configurations = bernoulli.configurations()

    assert len(configurations) == 19
    assert "Epsilon epsilon=0.1, decay=0.999" in configurations
    assert "Epsilon epsilon=0, optimistic" in configurations
    assert "Thompson Beta(0.5,0.5)" in configurations
    assert "KL-UCB c=3" in configurations
    assert "UCB-V(default)" in configurations


def test_drifting_scenario_includes_recency_weighted_configurations() -> None:
    drifting = next(s for s in SCENARIOS if s.key == "drifting")
    configurations = drifting.configurations()

    assert "Epsilon epsilon=0.1, alpha=0.1" in configurations
    assert "UCB SW(tau=1000)" in configurations
    assert "Thompson decay=0.95" in configurations
    assert drifting.reference_reward is None


def test_slow_drifting_scenario_its_own_instance() -> None:
    slow = next(s for s in SCENARIOS if s.key == "slow_drifting")
    configurations = slow.configurations()

    assert len(configurations) == 14
    assert "UCB SW(tau=4000)" in configurations
    assert "UCB SW(tau=1000)" in configurations
    assert slow.reference_reward is None
