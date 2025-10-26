"""Core Pydantic models for GOAP-based regression pipelines."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, cast

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


ModelFactory = Callable[[dict[str, Any]], BaseEstimator]


class ModelKind(StrEnum):
    """Enumeration of supported regression estimators."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"


def _linear_regression_factory(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**params)


def _random_forest_factory(params: dict[str, Any]) -> BaseEstimator:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params)


MODEL_FACTORIES: dict[ModelKind, ModelFactory] = {
    ModelKind.LINEAR_REGRESSION: _linear_regression_factory,
    ModelKind.RANDOM_FOREST: _random_forest_factory,
}


def _collect_supported_model_params() -> dict[ModelKind, frozenset[str]]:
    """Return a map of estimator kinds to the parameters they accept."""
    supported: dict[ModelKind, frozenset[str]] = {}
    for kind, factory in MODEL_FACTORIES.items():
        estimator = factory({})
        get_params = getattr(estimator, "get_params", None)
        if get_params is None:  # pragma: no cover - defensive guard
            message = f"Estimator for {kind} does not expose get_params()"
            raise TypeError(message)
        params = frozenset(str(name) for name in get_params())
        supported[kind] = params
    return supported


_SUPPORTED_MODEL_PARAMS = _collect_supported_model_params()


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

    kind: ModelKind = ModelKind.LINEAR_REGRESSION
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def ensure_string_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject parameter dictionaries with non-string keys."""
        keys: Iterable[object] = cast("Iterable[object]", value)
        invalid = [key for key in keys if not isinstance(key, str)]
        if invalid:
            joined = ", ".join(map(str, invalid))
            message = f"Model parameters must use string keys: {joined}"
            raise TypeError(message)
        return value

    @model_validator(mode="after")
    def validate_supported_parameters(self) -> ModelPolicy:
        """Ensure the provided parameters are accepted by the estimator."""
        supported = _SUPPORTED_MODEL_PARAMS.get(self.kind)
        if supported is None:  # pragma: no cover - defensive guard
            message = f"Unsupported model kind: {self.kind}"
            raise ValueError(message)

        unexpected = sorted(name for name in self.params if name not in supported)
        if unexpected:
            joined = ", ".join(unexpected)
            message = f"Unsupported parameter(s) for {self.kind}: {joined}"
            raise ValueError(message)
        return self


class EvalPolicy(BaseModel):
    """Enumerate the evaluation metrics to compute."""

    metrics: list[Literal["r2", "rmse", "mae"]] = Field(
        default_factory=lambda: ["r2", "rmse", "mae"],
    )


class ArtifactSpec(BaseModel):
    """Configure persistence for trained artefacts."""

    directory: str = Field(default="artifacts", min_length=1)
    model_filename: str = Field(default="model.joblib", min_length=1)
    preprocessor_filename: str = Field(default="preprocessor.joblib", min_length=1)

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, value: str) -> str:
        """Ensure the artefact directory is a safe relative path."""
        path = PurePath(value)
        if path.is_absolute():
            message = "Artifact directory must be a relative path."
            raise ValueError(message)
        if any(part == ".." for part in path.parts):
            message = "Artifact directory must not contain '..'."
            raise ValueError(message)
        return value

    @field_validator("model_filename", "preprocessor_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Ensure artefact filenames do not escape the target directory."""
        path = PurePath(value)
        if path.is_absolute():
            message = "Artifact filename must be a relative name."
            raise ValueError(message)
        if any(part == ".." for part in path.parts):
            message = "Artifact filename must not contain '..'."
            raise ValueError(message)
        if len(path.parts) != 1:
            message = "Artifact filename must not contain path separators."
            raise ValueError(message)
        return value

    def resolve_directory(self, root: Path) -> Path:
        """Return the artefact directory resolved under ``root``."""
        base = root.resolve()
        resolved = (base / PurePath(self.directory)).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:  # pragma: no cover - defensive guard
            message = "Artifact directory escapes the configured output root."
            raise ValueError(message) from exc
        return resolved


class PlannerPolicy(BaseModel):
    """Planner configuration for the GOAP A* search."""

    algorithm: Literal["astar"] = "astar"
    replan_on_failure: bool = True
    max_expansions: int = 10_000


class PipelineConfig(BaseModel):
    """Aggregate configuration passed to the GOAP execution engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts_root: Path = Field(default_factory=Path.cwd)
    file: FileSpec
    target: TargetSpec = TargetSpec()
    missing: MissingPolicy = MissingPolicy()
    category: CategoryPolicy = CategoryPolicy()
    scaling: ScalingPolicy = ScalingPolicy()
    split: SplitPolicy = SplitPolicy()
    model: ModelPolicy = ModelPolicy()
    eval: EvalPolicy = EvalPolicy()
    artifacts: ArtifactSpec = ArtifactSpec()
    planner: PlannerPolicy = PlannerPolicy()

    @model_validator(mode="after")
    def validate_artifact_directory(self) -> PipelineConfig:
        """Ensure artefact paths remain within the configured output root."""
        # Trigger resolution to validate the configuration eagerly.
        _ = self.resolve_artifact_directory()
        return self

    def resolve_artifact_directory(self) -> Path:
        """Return the directory where artefacts should be stored."""
        return self.artifacts.resolve_directory(Path(self.artifacts_root))

    def resolve_artifact_paths(self) -> tuple[Path, Path, Path]:
        """Return resolved paths for the artefact directory and files."""
        directory = self.resolve_artifact_directory()
        model_path = directory / self.artifacts.model_filename
        preprocessor_path = directory / self.artifacts.preprocessor_filename
        return directory, model_path, preprocessor_path


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
    model_path: Path | None = None
    preprocessor_path: Path | None = None

    def has(self, fact: str) -> bool:
        """Return whether the state currently holds a fact."""
        return fact in self.facts

    def add(self, fact: str) -> None:
        """Record that a fact now holds true."""
        self.facts.add(fact)

    def remove(self, fact: str) -> None:
        """Remove a fact when it is no longer valid."""
        self.facts.discard(fact)

