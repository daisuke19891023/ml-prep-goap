"""Unit tests for the GOAP pipeline configuration models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from goapml.models import FileSpec, ModelKind, ModelPolicy, PipelineConfig, SplitPolicy

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


def test_model_policy_rejects_unknown_parameters() -> None:
    """Model policies reject parameters outside the allow-list."""
    with pytest.raises(ValidationError) as exc:
        ModelPolicy(kind=ModelKind.LINEAR_REGRESSION, params={"alpha": 1.0})

    assert "Extra inputs are not permitted" in str(exc.value)


def test_model_policy_rejects_unsafe_parallelism() -> None:
    """Model policies refuse configurations with unsafe parameter values."""
    with pytest.raises(ValidationError) as exc:
        ModelPolicy(kind=ModelKind.RANDOM_FOREST, params={"n_jobs": -1})

    assert "Input should be greater than or equal to 1" in str(exc.value)

