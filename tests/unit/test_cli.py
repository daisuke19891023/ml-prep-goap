"""Unit tests for the GOAP ML CLI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from typer.testing import CliRunner

from goapml import cli
from goapml.cli import app


def test_cli_reports_directory_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should surface mkdir failures as a user-facing error."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        csv_path = Path("dataset.csv")
        csv_path.write_text("num,target\n1,2\n", encoding="utf-8")

        json_path = Path("output") / "result.json"

        def fake_execute_with_replanning(
            *_args: object, **_kwargs: object,
        ) -> list[str]:
            return ["evaluate"]

        monkeypatch.setattr(
            cli, "execute_with_replanning", fake_execute_with_replanning,
        )
        monkeypatch.setattr(cli.path_utils, "secure_open_supported", lambda: True)

        original_mkdir = cli.Path.mkdir

        def failing_mkdir(
            self: Path,
            mode: int = 0o777,
            parents: bool = False,
            exist_ok: bool = False,
        ) -> None:
            if self == json_path.parent or self.name == json_path.parent.name:
                raise OSError("boom")
            original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(cli.Path, "mkdir", failing_mkdir)

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
                "--json-out",
                str(json_path),
            ],
        )

    assert result.exit_code == 1
    assert "Unable to create directory for JSON output" in result.output
