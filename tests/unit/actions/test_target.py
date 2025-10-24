"""Tests for the IdentifyTarget action implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from goapml.actions.target import IdentifyTarget, ValidateTargetNumeric
from goapml.models import FileSpec, PipelineConfig, TargetSpec, WorldState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def pipeline_config(tmp_path: Path) -> Callable[[TargetSpec], PipelineConfig]:
    """Return a pipeline configuration backed by a temporary CSV file."""
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    file_spec = FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )

    def _build_config(target: TargetSpec) -> PipelineConfig:
        return PipelineConfig(file=file_spec, target=target)

    return _build_config


def build_state(frame: pd.DataFrame) -> WorldState:
    """Create a world state with the CSV already loaded."""
    return WorldState(df=frame, facts={"csv_loaded"})


def test_explicit_strategy_selects_configured_column(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Explicit strategy returns the provided column name when present."""
    frame = pd.DataFrame({"feature": [1], "target": [2]})
    config = pipeline_config(TargetSpec(strategy="explicit", name="target"))
    state = build_state(frame)

    IdentifyTarget().run(state, config)

    assert state.target == "target"
    assert state.has("target_identified")


def test_explicit_strategy_missing_column_raises(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Explicit strategy fails when the named column is absent."""
    frame = pd.DataFrame({"feature": [1], "other": [2]})
    config = pipeline_config(TargetSpec(strategy="explicit", name="target"))
    state = build_state(frame)

    message = "Target column 'target' not found in the dataset."
    with pytest.raises(ValueError, match=message):
        IdentifyTarget().run(state, config)


def test_validate_target_numeric_coerces_and_imputes(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Target column is coerced to numeric and sparse NaNs are imputed."""
    frame = pd.DataFrame(
        {"feature": [1, 2, 3, 4, 5], "target": ["1", "2", "bad", "4", "5"]},
    )
    config = pipeline_config(TargetSpec(strategy="explicit", name="target"))
    state = build_state(frame)

    IdentifyTarget().run(state, config)
    ValidateTargetNumeric().run(state, config)

    assert state.has("target_is_numeric")
    assert state.df is not None
    target_series = state.df["target"]
    assert is_numeric_dtype(target_series)
    assert target_series.iloc[2] == 3.0


def test_validate_target_numeric_fails_when_nan_ratio_high(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """High NaN ratios after coercion cause the action to fail with logging."""
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": ["bad", "worse", "3", "4", "bad"],
        },
    )
    config = pipeline_config(TargetSpec(strategy="explicit", name="target"))
    state = build_state(frame)

    IdentifyTarget().run(state, config)

    with pytest.raises(ValueError, match="could not be coerced"):
        ValidateTargetNumeric().run(state, config)

    assert state.logs[-1].startswith("target_numeric_failed:")


def test_heuristic_strategy_prefers_priority_names(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Heuristic strategy prioritises target, y, label, then *_y columns."""
    frame = pd.DataFrame({"feature": [1], "Target": [2], "foo_y": [3]})
    config = pipeline_config(TargetSpec(strategy="by_name_heuristic"))
    state = build_state(frame)

    IdentifyTarget().run(state, config)

    assert state.target == "Target"


def test_heuristic_strategy_falls_back_to_suffix(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """When priority names are missing, *_y columns are selected."""
    frame = pd.DataFrame({"feature": [1], "foo_y": [2], "bar": [3]})
    config = pipeline_config(TargetSpec(strategy="by_name_heuristic"))
    state = build_state(frame)

    IdentifyTarget().run(state, config)

    assert state.target == "foo_y"


def test_heuristic_strategy_without_matches_raises(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Heuristic strategy raises when no suitable column is found."""
    frame = pd.DataFrame({"feature": [1], "value": [2]})
    config = pipeline_config(TargetSpec(strategy="by_name_heuristic"))
    state = build_state(frame)

    message = "No suitable target column found via heuristics."
    with pytest.raises(ValueError, match=message):
        IdentifyTarget().run(state, config)


def test_last_column_strategy_selects_final_column(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Last-column strategy returns the final column from the dataset."""
    frame = pd.DataFrame({"x": [1], "y": [2]})
    config = pipeline_config(TargetSpec(strategy="last_column"))
    state = build_state(frame)

    IdentifyTarget().run(state, config)

    assert state.target == "y"


def test_last_column_strategy_empty_dataframe_raises(
    pipeline_config: Callable[[TargetSpec], PipelineConfig],
) -> None:
    """Empty DataFrames result in a validation error for last-column strategy."""
    frame = pd.DataFrame()
    config = pipeline_config(TargetSpec(strategy="last_column"))
    state = build_state(frame)

    message = "Cannot select target column from empty DataFrame."
    with pytest.raises(ValueError, match=message):
        IdentifyTarget().run(state, config)
