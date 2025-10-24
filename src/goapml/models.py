"""Core Pydantic models for GOAP-based regression pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator


_MIN_TEST_SIZE = 0.05
_MAX_TEST_SIZE = 0.5


if TYPE_CHECKING:
    import numpy as np
    from pandas import DataFrame, Series
    from sklearn.base import BaseEstimator, TransformerMixin
else:  # pragma: no cover - fallback types for runtime use only
    class _Float64(float):
        """Runtime placeholder for numpy float64 type."""

    class _NumpyStub:
        float64 = _Float64

    np = _NumpyStub()
    DataFrame = object
    Series = object
    BaseEstimator = object
    TransformerMixin = object


FeatureMatrix = Sequence[Sequence[float]] | DataFrame | NDArray[np.float64]
TargetVector = Sequence[float] | Series | NDArray[np.float64]
XYPair = tuple[FeatureMatrix, TargetVector]
TrainTestSplit = tuple[FeatureMatrix, FeatureMatrix, TargetVector, TargetVector]
PredictionVector = Sequence[float] | Series | NDArray[np.float64]


def _empty_fact_set() -> set[str]:
    return set()


def _empty_log_list() -> list[str]:
    return []


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


@dataclass(slots=True)
class WorldState:
    """Mutable container storing planner facts and runtime artefacts."""

    facts: set[str] = field(default_factory=_empty_fact_set)
    df: DataFrame | None = None
    encoding: str | None = None
    target: str | None = None
    xy: XYPair | None = None
    split: TrainTestSplit | None = None
    col_types: dict[str, Literal["numeric", "categorical"]] | None = None
    preprocessor: TransformerMixin | None = None
    model: BaseEstimator | None = None
    pred: PredictionVector | None = None
    metrics: dict[str, float] | None = None
    logs: list[str] = field(default_factory=_empty_log_list)

    def has(self, fact: str) -> bool:
        """Return whether the state currently holds a fact."""
        return fact in self.facts

    def add(self, fact: str) -> None:
        """Record that a fact now holds true."""
        self.facts.add(fact)

    def remove(self, fact: str) -> None:
        """Remove a fact when it is no longer valid."""
        self.facts.discard(fact)

