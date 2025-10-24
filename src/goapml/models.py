"""Core Pydantic models for GOAP-based regression pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_MIN_TEST_SIZE = 0.05
_MAX_TEST_SIZE = 0.5


class FileSpec(BaseModel):
    """Describe the CSV file that seeds the regression pipeline."""

    path: str = Field(..., description="CSVファイルのパス")
    encoding: str | None = Field(
        None, description="Explicit file encoding (auto-detected when omitted)",
    )
    delimiter: str = Field(",", min_length=1)
    decimal: str = Field(".", min_length=1)
    has_header: bool = True

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, value: str) -> str:
        """Ensure the configured CSV file is present on disk."""
        if not Path(value).is_file():
            message = f"CSV not found: {value}"
            raise ValueError(message)
        return value


class TargetSpec(BaseModel):
    """Describe how to locate and validate the regression target column."""

    strategy: Literal["explicit", "by_name_heuristic", "last_column"] = "explicit"
    name: str | None = None
    allowed_dtypes: list[str] = Field(default_factory=lambda: ["number"])


class MissingPolicy(BaseModel):
    """Configure handling of missing data for numeric and categorical features."""

    numeric: Literal["mean", "median", "most_frequent", "constant", "drop_rows"] = "median"
    categorical: Literal["most_frequent", "constant", "drop_rows"] = "most_frequent"
    fill_value_numeric: float | None = None
    fill_value_categorical: str | None = None
    report_threshold: float = 0.5


class CategoryPolicy(BaseModel):
    """Define categorical encoding behaviour."""

    encode: Literal["onehot", "ordinal", "none"] = "onehot"
    handle_unknown: Literal["ignore", "error"] = "ignore"


class ScalingPolicy(BaseModel):
    """Specify feature scaling strategy for numeric columns."""

    strategy: Literal["standard", "minmax", "robust", "none"] = "standard"


class SplitPolicy(BaseModel):
    """Describe how to split the dataset into training and validation sets."""

    test_size: float = 0.2
    random_state: int = 42
    stratify_by_target_quantiles: bool = False

    @field_validator("test_size")
    @classmethod
    def test_size_range(cls, value: float) -> float:
        """Validate that the configured split ratio is sensible."""
        if not _MIN_TEST_SIZE <= value < _MAX_TEST_SIZE:
            message = "test_size must be in [0.05, 0.5)."
            raise ValueError(message)
        return value


class ModelPolicy(BaseModel):
    """Describe which regression model to train and its parameters."""

    kind: Literal["linear_regression", "random_forest"] = "linear_regression"
    params: dict[str, Any] = Field(default_factory=dict)


class EvalPolicy(BaseModel):
    """Enumerate the evaluation metrics to compute."""

    metrics: list[Literal["r2", "rmse", "mae"]] = Field(
        default_factory=lambda: ["r2", "rmse", "mae"],
    )


class PlannerPolicy(BaseModel):
    """Planner configuration for the GOAP A* search."""

    algorithm: Literal["astar"] = "astar"
    replan_on_failure: bool = True
    max_expansions: int = 10_000


class PipelineConfig(BaseModel):
    """Aggregate configuration passed to the GOAP execution engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file: FileSpec
    target: TargetSpec = TargetSpec()
    missing: MissingPolicy = MissingPolicy()
    category: CategoryPolicy = CategoryPolicy()
    scaling: ScalingPolicy = ScalingPolicy()
    split: SplitPolicy = SplitPolicy()
    model: ModelPolicy = ModelPolicy()
    eval: EvalPolicy = EvalPolicy()
    planner: PlannerPolicy = PlannerPolicy()

