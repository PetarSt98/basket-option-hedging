"""Figures preserve the scheduled calendar and identify their saved financial inputs."""

import csv
from hashlib import sha256
import json
import math

import pytest

from basket_hedging.historical import POLICIES
from basket_hedging.historical_plots import draw_figures


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("versioned", [False, True])
def test_saved_figure_values_gap_markers_and_version_labels(tmp_path, monkeypatch, versioned):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    rows = []
    for date, error in (("2020-01-02", -10.), ("2020-02-03", 1000.), ("2020-03-02", 10.)):
        for policy in POLICIES:
            stopped = date == "2020-02-03" and policy == "volgan"
            rows.append(dict(episode=date, policy=policy, status="STOP_DATA_GAP" if stopped else "COMPLETED",
                             hedge_error=None if stopped else error))
    write_csv(tmp_path / "episodes.csv", rows)
    summary = dict(metrics=[dict(policy=policy, mean_turnover=100., mean_transaction_costs=2.,
                                 hedge_error_sd=math.sqrt(200.)) for policy in POLICIES])
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    representative = [dict(policy=policy, time=f"2020-01-0{day}T16:00:00+00:00", basket=100. + day,
                           liability_value=-50. + day, net_pnl=float(day), SPY=-1., QQQ=-2., IWM=-3.)
                      for policy in POLICIES for day in (2, 3)]
    write_csv(tmp_path / "representative.csv", representative)
    if versioned:
        (tmp_path / "benchmark_version.json").write_text(json.dumps(dict(
            label="Solver reliability correction; original frozen VolGAN, not a model improvement.",
            policy_labels={"volgan": "Original frozen VolGAN"})), encoding="utf-8")
    before = {path.name: sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    plotted = []
    plot = Axes.plot

    def observe_plot(self, *args, **kwargs):
        if len(args) >= 2 and len(args[0]) == 3 and "label" in kwargs:
            plotted.append(list(args[1]))
        return plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", observe_plot)
    paths = draw_figures(tmp_path)
    assert len(paths) == 8 and all(path.stat().st_size > 1000 for path in paths)
    data = json.loads((tmp_path / "figures" / "figure_data.json").read_text())
    assert data["input_sha256"] == before
    assert data["common_completed_episode_ids"] == ["2020-01-02", "2020-03-02"]
    assert data["excluded_episode_ids"] == ["2020-02-03"]
    assert data["excluded_statuses"]["2020-02-03"]["unhedged"] == "COMPLETED"
    assert data["excluded_statuses"]["2020-02-03"]["volgan"] == "STOP_DATA_GAP"
    assert all(values == [-10., None, 10.] for values in data["time"]["values"].values())
    assert all(values == [-10., 10.] for values in data["distribution"].values())
    assert len(plotted) == 5
    assert all(values[0] == -10. and math.isnan(values[1]) and values[2] == 10. for values in plotted)
    assert data["risk_cost"] == summary["metrics"]
    assert data["representative"]["episode"] == "2020-01-02"
    assert data["representative"]["rows"] == representative
    assert "Matched episodes: 2/3" in data["caption"]
    assert "Excluded episodes: 1. Failed runs: 1/15" in data["caption"]
    expected_label = "Original frozen VolGAN" if versioned else "VolGAN"
    assert data["policy_labels"]["volgan"] == expected_label
    svg = (tmp_path / "figures" / "02_error_through_time.svg").read_text(encoding="utf-8")
    assert "Excluded episodes (1)" in svg and expected_label in svg
    if versioned:
        assert "Solver reliability correction" in svg
    assert all(sha256((tmp_path / name).read_bytes()).hexdigest() == digest for name, digest in before.items())
