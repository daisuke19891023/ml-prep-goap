"""Integration tests for the Typer CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

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


def test_cli_run_creates_json_result() -> None:
    """The CLI should execute the pipeline and persist metrics to JSON."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        csv_path = Path("dataset.csv")
        json_path = Path("result.json")
        _write_sample_csv(csv_path)

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

        if os.name != "nt":
            mode = stat.S_IMODE(json_path.stat().st_mode)
            assert mode == 0o600


def test_cli_run_rejects_parent_directory_escape() -> None:
    """json-out paths escaping the working directory should be rejected."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        csv_path = Path("dataset.csv")
        _write_sample_csv(csv_path)
        escape_target = Path("..") / "escape.json"

        result = runner.invoke(
            app,
            [
                "run",
                "--csv",
                str(csv_path),
                "--target",
                "target",
                "--json-out",
                str(escape_target),
            ],
        )

        assert result.exit_code != 0
        assert "parent directory references" in result.output


def test_cli_run_rejects_symlink_output() -> None:
    """json-out paths pointing to symlinks should be rejected."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        csv_path = Path("dataset.csv")
        _write_sample_csv(csv_path)

        target = Path("real.json")
        target.write_text("{}", encoding="utf-8")
        link = Path("link.json")

        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symbolic links are not supported on this platform")

        result = runner.invoke(
            app,
            [
                "run",
                "--csv",
                str(csv_path),
                "--target",
                "target",
                "--json-out",
                str(link),
            ],
        )

        assert result.exit_code != 0
        assert "symbolic links" in result.output
