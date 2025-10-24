"""Actions responsible for preprocessing diagnostics and transformations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from goapml.schemas import (
    Action,
    ActionSchema,
    BUILD_PREPROCESSOR_SCHEMA,
    CHECK_MISSING_SCHEMA,
    FIT_TRANSFORM_PREPROCESSOR_SCHEMA,
)

if TYPE_CHECKING:
    from pandas import DataFrame

    from goapml.models import PipelineConfig, WorldState
    from numpy.typing import NDArray

__all__ = [
    "BUILD_PREPROCESSOR_SCHEMA",
    "CHECK_MISSING_SCHEMA",
    "FIT_TRANSFORM_PREPROCESSOR_SCHEMA",
    "BuildPreprocessor",
    "CheckMissing",
    "FitTransformPreprocessor",
]


_LOGGER = logging.getLogger(__name__)


_EXPECTED_FEATURE_DIMENSION = 2


class _TransformerProtocol(Protocol):
    """Structural protocol for preprocessors supporting fit/transform."""

    def fit(self, x: Any, y: Any | None = None) -> Any: ...

    def transform(self, x: Any) -> Any: ...


@dataclass(slots=True)
class CheckMissing(Action):
    """Compute column-wise missing ratios and report high-risk features."""

    schema: ActionSchema = field(default_factory=lambda: CHECK_MISSING_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Log missing value statistics and emit warnings for high ratios."""
        if state.xy is None:
            message = "Features and target must be separated before checking missingness."
            raise RuntimeError(message)

        features, _ = state.xy
        if not hasattr(features, "isna"):
            message = "Feature matrix must be a pandas DataFrame for missing checks."
            raise TypeError(message)

        frame = cast("DataFrame", features)
        total_rows = len(frame)

        if total_rows == 0:
            missing_ratios = pd.Series(0.0, index=frame.columns, dtype=float)
        else:
            missing_counts = frame.isna().sum()
            missing_ratios = missing_counts.astype(float) / float(total_rows)

        sorted_ratios = sorted(
            ((str(column), float(ratio)) for column, ratio in missing_ratios.items()),
            key=lambda item: (-item[1], item[0]),
        )
        top_five = sorted_ratios[:5]
        formatted = (
            ",".join(f"{column}:{ratio:.3f}" for column, ratio in top_five)
            if top_five
            else "none"
        )
        state.logs.append(f"missing_top5:{formatted}")

        threshold = config.missing.report_threshold
        exceeding = [item for item in sorted_ratios if item[1] > threshold]
        if exceeding:
            warning_details = ", ".join(f"{column}:{ratio:.3f}" for column, ratio in exceeding)
            _LOGGER.warning(
                "Missing ratios above threshold %.2f: %s", threshold, warning_details,
            )

        state.add("missing_checked")


@dataclass(slots=True)
class BuildPreprocessor(Action):
    """Construct the preprocessing pipeline according to the configuration."""

    schema: ActionSchema = field(default_factory=lambda: BUILD_PREPROCESSOR_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Instantiate a column transformer based on column types."""
        if state.split is None or state.col_types is None:
            message = "Train/test split must be completed before building the preprocessor."
            raise RuntimeError(message)

        x_train, _, _, _ = state.split

        if not hasattr(x_train, "columns"):
            message = "Training features must be a pandas DataFrame to build the preprocessor."
            raise TypeError(message)

        numeric_columns = [
            column
            for column, kind in state.col_types.items()
            if kind == "numeric"
        ]
        categorical_columns = [
            column
            for column, kind in state.col_types.items()
            if kind == "categorical"
        ]

        if not numeric_columns and not categorical_columns:
            message = "No feature columns available to build a preprocessor."
            raise ValueError(message)

        transformers: list[tuple[str, Any, list[str]]] = []

        if numeric_columns:
            numeric_transformer = self._build_numeric_pipeline(config)
            transformers.append(("numeric", numeric_transformer, numeric_columns))

        if categorical_columns:
            categorical_transformer = self._build_categorical_pipeline(config)
            transformers.append(("categorical", categorical_transformer, categorical_columns))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
        )

        state.preprocessor = preprocessor
        state.add("preprocessor_built")
        state.logs.append(
            "build_preprocessor:"
            f"num={len(numeric_columns)},cat={len(categorical_columns)}",
        )

    def _build_numeric_pipeline(self, config: PipelineConfig) -> Pipeline:
        strategy = config.missing.numeric
        if strategy == "drop_rows":
            message = "drop_rows missing policy is not supported for numeric features."
            raise ValueError(message)

        imputer_kwargs: dict[str, Any] = {}
        if strategy == "constant":
            fill_value = config.missing.fill_value_numeric
            imputer_kwargs["fill_value"] = 0.0 if fill_value is None else float(fill_value)

        numeric_imputer = SimpleImputer(strategy=strategy, **imputer_kwargs)

        scaler = self._select_scaler(config)
        steps: list[tuple[str, Any]] = [("impute", numeric_imputer)]
        if scaler is not None:
            steps.append(("scale", scaler))
        return Pipeline(steps)

    def _build_categorical_pipeline(self, config: PipelineConfig) -> Pipeline:
        strategy = config.missing.categorical
        if strategy == "drop_rows":
            message = "drop_rows missing policy is not supported for categorical features."
            raise ValueError(message)

        imputer_kwargs: dict[str, Any] = {}
        if strategy == "constant":
            fill_value = config.missing.fill_value_categorical or "missing"
            imputer_kwargs["fill_value"] = fill_value

        categorical_imputer = SimpleImputer(
            strategy="most_frequent" if strategy != "constant" else "constant",
            **imputer_kwargs,
        )

        encoder = self._select_encoder(config)
        steps: list[tuple[str, Any]] = [("impute", categorical_imputer)]
        if encoder is not None:
            steps.append(("encode", encoder))
        return Pipeline(steps)

    @staticmethod
    def _select_scaler(
        config: PipelineConfig,
    ) -> StandardScaler | MinMaxScaler | RobustScaler | None:
        strategy = config.scaling.strategy
        if strategy == "standard":
            return StandardScaler()
        if strategy == "minmax":
            return MinMaxScaler()
        if strategy == "robust":
            return RobustScaler()
        if strategy == "none":
            return None
        message = f"Unsupported scaling strategy: {strategy}"
        raise ValueError(message)

    @staticmethod
    def _select_encoder(config: PipelineConfig) -> OneHotEncoder | OrdinalEncoder | None:
        encode = config.category.encode
        handle_unknown = config.category.handle_unknown
        if encode == "onehot":
            return OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=False,
                dtype=np.float64,
            )
        if encode == "ordinal":
            if handle_unknown == "ignore":
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            return OrdinalEncoder()
        if encode == "none":
            return None
        message = f"Unsupported categorical encoding strategy: {encode}"
        raise ValueError(message)


@dataclass(slots=True)
class FitTransformPreprocessor(Action):
    """Fit a preprocessing transformer and apply it to train/test splits."""

    schema: ActionSchema = field(default_factory=lambda: FIT_TRANSFORM_PREPROCESSOR_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:  # noqa: ARG002
        """Fit the preprocessor on training data and transform both splits."""
        if state.preprocessor is None:
            message = "A preprocessor must be built before fitting and transforming."
            raise RuntimeError(message)
        if state.split is None:
            message = "Train/test split must be available before preprocessing."
            raise RuntimeError(message)

        x_train, x_test, y_train, y_test = state.split
        preprocessor = cast("_TransformerProtocol", state.preprocessor)

        try:
            preprocessor.fit(x_train, y_train)
        except TypeError:
            preprocessor.fit(x_train)

        x_train_processed = preprocessor.transform(x_train)
        x_test_processed = preprocessor.transform(x_test)

        train_array = self._as_2d_array(x_train_processed)
        test_array = self._as_2d_array(x_test_processed)

        if train_array.shape[1] != test_array.shape[1]:
            message = "Transformed feature matrices must share the same number of columns."
            raise RuntimeError(message)

        if train_array.shape[0] != len(y_train) or test_array.shape[0] != len(y_test):
            message = "Transformed feature rows must align with target lengths."
            raise RuntimeError(message)

        state.split = (train_array, test_array, y_train, y_test)
        state.add("features_ready")
        state.logs.append("fit_transform_preprocessor")

    @staticmethod
    def _as_2d_array(data: Any) -> NDArray[np.float64]:
        if hasattr(data, "toarray"):
            data = data.toarray()
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = np.reshape(array, (-1, 1))
        if array.ndim != _EXPECTED_FEATURE_DIMENSION:
            message = "Transformed data must be two-dimensional."
            raise RuntimeError(message)
        return cast("NDArray[np.float64]", array)

