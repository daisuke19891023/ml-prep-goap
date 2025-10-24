from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from goapml.actions.preprocess import CheckMissing
from goapml.models import FileSpec, MissingPolicy, PipelineConfig, WorldState

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _build_config(tmp_path: Path) -> PipelineConfig:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    file_spec = FileSpec(
        path=str(csv_path),
        encoding="utf-8",
        delimiter=",",
        decimal=".",
        has_header=True,
    )
    missing_policy = MissingPolicy(report_threshold=0.6)
    return PipelineConfig(file=file_spec, missing=missing_policy)


def test_check_missing_logs_top_five_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The action should log top missing columns and warn when threshold exceeded."""
    config = _build_config(tmp_path)
    features = pd.DataFrame(
        {
            "c0": [1.0, None, 3.0, None],
            "c1": [1.0, None, None, 4.0],
            "c2": [None, None, None, 4.0],
            "c3": [None, None, None, None],
            "c4": [1.0, None, 3.0, 4.0],
            "c5": [1.0, 2.0, 3.0, 4.0],
        },
    )
    target = pd.Series([1.0, 2.0, 3.0, 4.0])
    state = WorldState(xy=(features, target), facts={"xy_separated"})

    with caplog.at_level(logging.WARNING):
        CheckMissing().run(state, config)

    assert state.has("missing_checked")
    assert state.logs[-1] == "missing_top5:c3:1.000,c2:0.750,c0:0.500,c1:0.500,c4:0.250"

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    warning_message = warnings[0].message
    assert "c3:1.000" in warning_message
    assert "c2:0.750" in warning_message


def test_check_missing_no_warning_when_below_threshold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning should be emitted when all columns are below the threshold."""
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    file_spec = FileSpec(
        path=str(csv_path),
        encoding="utf-8",
        delimiter=",",
        decimal=".",
        has_header=True,
    )
    missing_policy = MissingPolicy(report_threshold=0.8)
    config = PipelineConfig(file=file_spec, missing=missing_policy)

    features = pd.DataFrame(
        {
            "c0": [1.0, None, 3.0, 4.0],
            "c1": [1.0, 2.0, None, 4.0],
            "c2": [1.0, 2.0, 3.0, 4.0],
        },
    )
    target = pd.Series([1.0, 2.0, 3.0, 4.0])
    state = WorldState(xy=(features, target), facts={"xy_separated"})

    with caplog.at_level(logging.WARNING):
        CheckMissing().run(state, config)

    assert state.logs[-1] == "missing_top5:c0:0.250,c1:0.250,c2:0.000"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == []
