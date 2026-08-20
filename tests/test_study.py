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


def test_bernoulli_scenario_sweeps_the_full_grid() -> None:
    bernoulli = next(s for s in SCENARIOS if s.key == "bernoulli")
    configurations = bernoulli.configurations()

    assert len(configurations) == 17
    assert "Epsilon epsilon=0.1, decay=0.999" in configurations
    assert "Epsilon epsilon=0, optimistic" in configurations
    assert "Thompson Beta(0.5,0.5)" in configurations


def test_drifting_scenario_includes_recency_weighted_configurations() -> None:
    drifting = next(s for s in SCENARIOS if s.key == "drifting")
    configurations = drifting.configurations()

    assert "Epsilon epsilon=0.1, alpha=0.1" in configurations
    assert drifting.reference_reward is None
