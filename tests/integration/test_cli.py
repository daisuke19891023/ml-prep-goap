"""Integration tests for the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from goapml.cli import app


def _write_sample_csv(path: Path) -> None:
    path.write_text(
        "num,cat,target\n"
        "1.0,a,10.0\n"
        "2.0,b,20.0\n"
        "3.0,a,30.0\n"
        "4.0,b,40.0\n",
        encoding="utf-8",
    )


def test_cli_run_creates_json_result(tmp_path: Path) -> None:
    """The CLI should execute the pipeline and persist metrics to JSON."""
    base_dir = Path(tmp_path)
    csv_path = base_dir / "dataset.csv"
    json_path = base_dir / "result.json"
    _write_sample_csv(csv_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--csv",
            str(csv_path),
            "--target",
            "target",
            "--model",
            "linear_regression",
            "--metrics",
            "r2",
            "rmse",
            "--json-out",
            str(json_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert "r2" in payload["metrics"]
    assert payload["logs"], "expected logs to be recorded"
    assert payload["actions"][-1] == "evaluate"
