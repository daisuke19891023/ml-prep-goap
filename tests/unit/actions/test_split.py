"""Tests for the SplitXY action implementation."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

from goapml.actions.split import SplitXY, TrainTestSplit
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


def test_train_test_split_records_split_and_fact(pipeline_config: PipelineConfig) -> None:
    """The train/test split action records split artefacts and facts."""
    frame = pd.DataFrame(
        {
            "num": list(range(10)),
            "cat": ["a", "b"] * 5,
            "target": [float(value) for value in range(10)],
        },
    )
    state = build_state(frame, "target")
    SplitXY().run(state, pipeline_config)

    TrainTestSplit().run(state, pipeline_config)

    assert state.split is not None
    x_train, x_test, y_train, y_test = state.split
    assert len(x_train) + len(x_test) == len(frame)
    assert len(y_train) + len(y_test) == len(frame)
    test_ratio = len(x_test) / len(frame)
    assert abs(test_ratio - pipeline_config.split.test_size) < 0.15
    assert state.has("split_done")
    assert state.logs[-1] == "train_test_split"


def test_train_test_split_quantile_stratification(pipeline_config: PipelineConfig) -> None:
    """Quantile stratification keeps representation from each quantile bin."""
    config = pipeline_config.model_copy()
    config.split = pipeline_config.split.model_copy(
        update={"stratify_by_target_quantiles": True},
    )

    frame = pd.DataFrame(
        {
            "num": list(range(100)),
            "cat": ["a", "b", "c", "d", "e"] * 20,
            "target": [float(value) for value in range(100)],
        },
    )
    state = build_state(frame, "target")
    SplitXY().run(state, config)

    TrainTestSplit().run(state, config)

    assert state.split is not None
    _, _, _, y_test = state.split

    y_test_series = cast("pd.Series", y_test)
    quantiles = frame["target"].quantile([0.2, 0.4, 0.6, 0.8])
    edges = [frame["target"].min() - 1e-9, *quantiles.tolist(), frame["target"].max() + 1e-9]

    for start, end in pairwise(edges):
        bin_mask = (y_test_series >= start) & (y_test_series <= end)
        assert bin_mask.any()
