"""End-to-end regression pipeline tests using canonical sklearn datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest
from pandas import DataFrame, Series
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

from goapml.engine import execute_with_replanning
from goapml.models import (
    EvalPolicy,
    FileSpec,
    ModelPolicy,
    PipelineConfig,
    SplitPolicy,
    TargetSpec,
    WorldState,
)
from goapml.schemas import Goal


def _run_pipeline(
    *,
    csv_path: Path,
    target: str,
    model_policy: ModelPolicy,
    split_policy: SplitPolicy | None = None,
    metrics: list[Literal["r2", "rmse", "mae"]] | None = None,
) -> dict[str, float]:
    """Execute the regression pipeline for the provided CSV dataset."""
    if metrics is None:
        metric_list: list[Literal["r2", "rmse", "mae"]] = ["r2", "rmse", "mae"]
    else:
        metric_list = metrics
    config = PipelineConfig(
        file=FileSpec(path=str(csv_path), encoding=None, delimiter=",", decimal="."),
        target=TargetSpec(strategy="explicit", name=target),
        split=split_policy or SplitPolicy(),
        model=model_policy,
        eval=EvalPolicy(metrics=metric_list),
    )
    state = WorldState(facts={"file_exists"})
    goal = Goal(required={"evaluated"})
    execute_with_replanning(state=state, config=config, goal=goal)
    metrics_dict = state.metrics
    assert metrics_dict is not None, "expected evaluation metrics to be recorded"
    return metrics_dict


@pytest.mark.integration
def test_end_to_end_synthetic_csv() -> None:
    """Pipeline should achieve strong performance on the bundled synthetic dataset."""
    data_path = Path("tests/data/tiny_synthetic.csv")
    assert data_path.is_file(), "synthetic CSV must be generated as part of the repository"

    model = ModelPolicy(kind="linear_regression", params={})
    metrics = _run_pipeline(csv_path=data_path, target="target", model_policy=model)

    assert metrics["r2"] >= 0.8


@pytest.mark.integration
def test_end_to_end_wine_label_regression(tmp_path: Path) -> None:
    """Predicting wine class labels as a regression problem should yield high R²."""
    raw_features, raw_target = cast(
        tuple[DataFrame, Series],
        load_wine(as_frame=True, return_X_y=True),
    )
    features = raw_features.copy()
    target_series: Series = raw_target
    frame = features.copy()
    frame[target_series.name or "target"] = target_series
    csv_path = tmp_path / "wine.csv"
    frame.to_csv(csv_path, index=False)

    model = ModelPolicy(
        kind="random_forest",
        params={"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    )
    metrics = _run_pipeline(csv_path=csv_path, target="target", model_policy=model)

    assert metrics["r2"] >= 0.9


@pytest.mark.integration
def test_end_to_end_breast_cancer_label_regression(tmp_path: Path) -> None:
    """Breast cancer labels treated as regression should remain highly predictable."""
    raw_features, raw_target = cast(
        tuple[DataFrame, Series],
        load_breast_cancer(as_frame=True, return_X_y=True),
    )
    features = raw_features.copy()
    target_series: Series = raw_target
    frame = features.copy()
    frame[target_series.name or "target"] = target_series
    csv_path = tmp_path / "breast_cancer.csv"
    frame.to_csv(csv_path, index=False)

    model = ModelPolicy(
        kind="random_forest",
        params={"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    )
    metrics = _run_pipeline(csv_path=csv_path, target="target", model_policy=model)

    assert metrics["r2"] >= 0.8


@pytest.mark.integration
def test_end_to_end_digits_label_regression(tmp_path: Path) -> None:
    """Digit labels formulated as a regression task should surpass the success threshold."""
    raw_features, raw_target = cast(
        tuple[DataFrame, Series],
        load_digits(as_frame=True, return_X_y=True),
    )
    features = raw_features.copy()
    target_series: Series = raw_target
    frame = features.copy()
    frame[target_series.name or "target"] = target_series
    csv_path = tmp_path / "digits.csv"
    frame.to_csv(csv_path, index=False)

    model = ModelPolicy(
        kind="random_forest",
        params={"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    )
    metrics = _run_pipeline(csv_path=csv_path, target="target", model_policy=model)

    assert metrics["r2"] >= 0.85
