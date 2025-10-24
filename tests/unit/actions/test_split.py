"""Tests for the SplitXY action implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from goapml.actions.split import SplitXY
from goapml.models import FileSpec, PipelineConfig, TargetSpec, WorldState


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Build a minimal pipeline configuration for the split action."""
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("num,cat,target\n1.0,a,10\n", encoding="utf-8")

    file_spec = FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )

    target_spec = TargetSpec(strategy="explicit", name="target")
    return PipelineConfig(file=file_spec, target=target_spec)


def build_state(frame: pd.DataFrame, target: str) -> WorldState:
    """Return a world state with a numeric-validated target column."""
    return WorldState(df=frame.copy(), facts={"target_is_numeric"}, target=target)


def test_split_xy_assigns_xy_and_types(pipeline_config: PipelineConfig) -> None:
    """Features and target are separated and column types inferred."""
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "target": [10.0, 20.0, 30.0],
        },
    )
    state = build_state(frame, "target")

    SplitXY().run(state, pipeline_config)

    assert state.xy is not None
    features_obj, target_obj = state.xy
    assert isinstance(features_obj, pd.DataFrame)
    assert isinstance(target_obj, pd.Series)
    features = features_obj
    target = target_obj
    assert list(features.columns) == ["num", "cat"]
    assert target.name == "target"
    assert state.col_types == {"num": "numeric", "cat": "categorical"}
    assert state.has("xy_separated")
    assert state.logs[-1] == "split_xy"
