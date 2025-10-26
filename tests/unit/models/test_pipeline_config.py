"""Unit tests for the GOAP pipeline configuration models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

import goapml.models
from goapml.models import (
    FileSpec,
    ModelKind,
    ModelPolicy,
    PipelineConfig,
    SplitPolicy,
    get_supported_model_params,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_pipeline_config_accepts_existing_csv(tmp_path: Path) -> None:
    """`PipelineConfig` succeeds when the referenced CSV exists."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    config = PipelineConfig(
        file=FileSpec(
            path=str(csv_path),
            encoding=None,
            delimiter=",",
            decimal=".",
            has_header=True,
        ),
    )

    assert config.file.path == str(csv_path)
    assert config.target.strategy == "explicit"


def test_pipeline_config_rejects_missing_csv(tmp_path: Path) -> None:
    """`PipelineConfig` raises a validation error for a missing CSV."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=FileSpec(
                path=str(missing_path),
                encoding=None,
                delimiter=",",
                decimal=".",
                has_header=True,
            ),
        )

    assert "CSV not found" in str(exc.value)


def test_split_policy_enforces_reasonable_test_size(tmp_path: Path) -> None:
    """`SplitPolicy` rejects configurations with extreme test_size ratios."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=FileSpec(
                path=str(csv_path),
                encoding=None,
                delimiter=",",
                decimal=".",
                has_header=True,
            ),
            split=SplitPolicy(test_size=0.9),
        )

    assert "test_size must be in [0.05, 0.5)." in str(exc.value)


def test_model_policy_rejects_unknown_parameters(tmp_path: Path) -> None:
    """`ModelPolicy` should reject parameters not supported by the estimator."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=FileSpec(
                path=str(csv_path),
                encoding=None,
                delimiter=",",
                decimal=".",
                has_header=True,
            ),
            model=ModelPolicy(
                kind=ModelKind.LINEAR_REGRESSION,
                params={"unsupported": True},
            ),
        )

    assert "Unsupported parameter(s) for linear_regression" in str(exc.value)


def test_model_policy_rejects_non_string_parameter_keys(tmp_path: Path) -> None:
    """`ModelPolicy` should reject parameter dictionaries with non-string keys."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=FileSpec(
                path=str(csv_path),
                encoding=None,
                delimiter=",",
                decimal=".",
                has_header=True,
            ),
            model=ModelPolicy(
                kind=ModelKind.RANDOM_FOREST,
                params=cast("dict[str, Any]", {1: "invalid"}),
            ),
        )

    assert "Input should be a valid string" in str(exc.value)


def test_supported_model_params_uses_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ModelPolicy` should build allow-lists from schemas without estimator instantiation."""
    calls: list[ModelKind] = []

    class _SentinelEstimator:
        def get_params(self) -> dict[str, Any]:  # pragma: no cover - fallback path
            return {}

    def _recording_factory(_: dict[str, Any]) -> _SentinelEstimator:
        calls.append(ModelKind.LINEAR_REGRESSION)
        return _SentinelEstimator()

    def _recording_factory_rf(_: dict[str, Any]) -> _SentinelEstimator:
        calls.append(ModelKind.RANDOM_FOREST)
        return _SentinelEstimator()

    monkeypatch.setitem(
        goapml.models.MODEL_FACTORIES, ModelKind.LINEAR_REGRESSION, _recording_factory,
    )
    monkeypatch.setitem(
        goapml.models.MODEL_FACTORIES, ModelKind.RANDOM_FOREST, _recording_factory_rf,
    )

    get_supported_model_params.cache_clear()

    ModelPolicy(kind=ModelKind.LINEAR_REGRESSION, params={"fit_intercept": True})

    assert calls == []

    get_supported_model_params.cache_clear()

