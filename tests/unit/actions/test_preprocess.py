from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from goapml.actions.preprocess import CheckMissing, FitTransformPreprocessor
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


def test_fit_transform_preprocessor_returns_numpy_arrays(tmp_path: Path) -> None:
    """The preprocessor should fit on train data and return dense arrays."""
    config = _build_config(tmp_path)
    features = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "c", "b"],
        },
    )
    target = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)

    x_train = features.iloc[:3]
    x_test = features.iloc[3:]
    y_train = target.iloc[:3]
    y_test = target.iloc[3:]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["num"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["cat"],
            ),
        ],
    )

    state = WorldState(
        split=(x_train, x_test, y_train, y_test),
        preprocessor=preprocessor,
        facts={"preprocessor_built"},
    )

    FitTransformPreprocessor().run(state, config)

    assert state.split is not None
    assert state.preprocessor is not None
    assert isinstance(state.preprocessor, ColumnTransformer)
    x_train_proc, x_test_proc, y_train_proc, y_test_proc = state.split

    assert isinstance(x_train_proc, np.ndarray)
    assert isinstance(x_test_proc, np.ndarray)
    assert x_train_proc.dtype == np.float64
    assert x_test_proc.dtype == np.float64

    assert x_train_proc.shape[0] == len(x_train)
    assert x_test_proc.shape[0] == len(x_test)
    assert x_train_proc.shape[1] == x_test_proc.shape[1]

    assert y_train_proc is y_train
    assert y_test_proc is y_test

    assert state.has("features_ready")
    assert state.logs[-1] == "fit_transform_preprocessor"

    numeric_mean = float(np.mean(x_train_proc[:, 0]))
    assert np.isclose(numeric_mean, 0.0, atol=1e-9)
