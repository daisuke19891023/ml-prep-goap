"""Actions responsible for preprocessing diagnostics and transformations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:
    from pandas import DataFrame

    from goapml.models import PipelineConfig, WorldState
    from numpy.typing import NDArray

__all__ = [
    "CHECK_MISSING_SCHEMA",
    "FIT_TRANSFORM_PREPROCESSOR_SCHEMA",
    "CheckMissing",
    "FitTransformPreprocessor",
]


_LOGGER = logging.getLogger(__name__)


_EXPECTED_FEATURE_DIMENSION = 2


class _TransformerProtocol(Protocol):
    """Structural protocol for preprocessors supporting fit/transform."""

    def fit(self, x: Any, y: Any | None = None) -> Any: ...

    def transform(self, x: Any) -> Any: ...


_CHECK_MISSING_SCHEMA = ActionSchema(
    name="check_missing",
    requires={"xy_separated"},
    provides={"missing_checked"},
    cost=1.0,
)

CHECK_MISSING_SCHEMA = _CHECK_MISSING_SCHEMA


@dataclass(slots=True)
class CheckMissing(Action):
    """Compute column-wise missing ratios and report high-risk features."""

    schema: ActionSchema = field(default_factory=lambda: _CHECK_MISSING_SCHEMA)

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


FIT_TRANSFORM_PREPROCESSOR_SCHEMA = ActionSchema(
    name="fit_transform_preprocessor",
    requires={"preprocessor_built"},
    provides={"features_ready"},
    cost=1.0,
)


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
