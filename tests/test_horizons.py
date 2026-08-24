"""Smoke tests for the regret-versus-horizon sweep."""

from study.horizons import run_horizon_sweep, write_horizon_results


def test_horizon_sweep_runs_and_writes_csv(tmp_path) -> None:
    results = run_horizon_sweep(horizons=(20, 40), seeds=(0, 1))
    path = write_horizon_results(results, seeds=(0, 1), output_dir=tmp_path)

    assert set(results) == {20, 40}
    assert path.is_file()
    header = path.read_text().splitlines()[0]
    for column in ("mean_regret", "median_regret", "regret_per_step"):
        assert column in header
    lines = path.read_text().splitlines()[1:]
    # one row per configuration and horizon
    assert len(lines) == 2 * len(results[20].summaries())
