import json
from pathlib import Path
import subprocess
import sys


def command(*arguments):
    return subprocess.run([sys.executable, "-m", "basket_hedging", *map(str, arguments)],
                          capture_output=True, text=True)


def test_installed_cli_runs_both_policies_and_preserves_breaches(tmp_path):
    fixtures = tmp_path / "fixtures"
    generated = command("make-deterministic", "--output-dir", fixtures)
    assert generated.returncode == 0, generated.stderr
    completed = command("run", "--case", fixtures / "deterministic_complete.json", "--policy", "both",
                        "--output", tmp_path / "complete")
    assert completed.returncode == 0, completed.stderr
    comparison = json.loads((tmp_path / "complete" / "comparison.json").read_text())
    assert {row["policy"]: row["marked_pnl"] for row in comparison["outcomes"]} == {
        "unhedged": -200, "delta": -371}
    breach = command("run", "--case", fixtures / "deterministic_gap.json", "--policy", "both",
                     "--output", tmp_path / "gap")
    assert breach.returncode == 0, breach.stderr
    comparison = json.loads((tmp_path / "gap" / "comparison.json").read_text())
    assert len(comparison["outcomes"]) == 2
    assert all(row["status"] == "STOP_WITH_BREACH" for row in comparison["outcomes"])
    assert all(row["terminal_pnl"] is None for row in comparison["outcomes"])


def test_cli_reports_malformed_input_as_error(tmp_path):
    case = tmp_path / "bad.json"
    case.write_text('{"schema_version": 999}')
    result = command("run", "--case", case, "--output", tmp_path / "reports")
    assert result.returncode == 2
    assert "unsupported case schema" in result.stderr
    assert not (tmp_path / "reports").exists()


def test_cli_synthetic_generation_uses_normal_saved_input(tmp_path):
    path = tmp_path / "synthetic.json"
    generated = command("generate", "--output", path, "--days", "2", "--samples", "128")
    assert generated.returncode == 0, generated.stderr
    assert path.exists() and path.with_suffix(".jsonl").exists()
    result = command("run", "--case", path, "--policy", "both", "--output", tmp_path / "reports")
    assert result.returncode == 0, result.stderr
    assert "unhedged: COMPLETED" in result.stdout
    assert "delta: COMPLETED" in result.stdout


def test_cli_runs_opt_in_scenario_comparison(tmp_path):
    path = tmp_path / "synthetic.json"
    generated = command("generate", "--output", path, "--days", "2", "--samples", "64")
    assert generated.returncode == 0, generated.stderr
    result = command("compare-scenarios", "--case", path, "--output", tmp_path / "research",
                     "--scenarios", "16", "--pricing-samples", "64")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "research" / "research_summary.json").read_text())
    assert [row["policy"] for row in report["outcomes"]] == ["unhedged", "delta", "variance", "cont_vuletic_l1"]
    assert all(row["status"] == "COMPLETED" for row in report["outcomes"])
