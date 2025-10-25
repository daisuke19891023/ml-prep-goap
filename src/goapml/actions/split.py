"""Actions handling feature/target separation and dataset splitting."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from goapml.schemas import (
    Action,
    ActionSchema,
    SPLIT_XY_SCHEMA,
    TRAIN_TEST_SPLIT_SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from goapml.models import (
        PipelineConfig,
        TargetVector,
        TrainTestSplit as TrainTestSplitTuple,
        WorldState,
    )
    from numpy.typing import NDArray

__all__ = ["SplitXY", "TrainTestSplit"]

_DEFAULT_QUANTILE_BINS = 5
_MIN_STRATIFY_CLASSES = 2


_LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class SplitXY(Action):
    """Split the dataframe into features and target while inferring column types."""

    schema: ActionSchema = field(default_factory=lambda: SPLIT_XY_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:  # noqa: ARG002
        """Separate X/y and determine feature column categories."""
        if state.df is None or state.target is None:
            message = "Target must be validated as numeric before splitting."
            raise RuntimeError(message)

        _LOGGER.info(
            "Separating features and target.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "split_xy",
                "columns": len(state.df.columns),
            },
        )
        features = state.df.drop(columns=[state.target])
        target = state.df[state.target]

        state.xy = (features, target)
        state.col_types = {
            column: "numeric" if is_numeric_dtype(features[column]) else "categorical"
            for column in features.columns
        }
        state.add("xy_separated")
        state.logs.append("split_xy")
        _LOGGER.info(
            "Feature/target separation complete.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "split_xy",
                "features": len(features.columns),
            },
        )


@dataclass(slots=True)
class TrainTestSplit(Action):
    """Split the separated features and target into train/test sets."""

    schema: ActionSchema = field(default_factory=lambda: TRAIN_TEST_SPLIT_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Perform the configured train/test split and record it on the state."""
        if state.xy is None:
            message = "Features and target must be separated before splitting."
            raise RuntimeError(message)

        _LOGGER.info(
            "Performing train/test split.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "train_test_split",
                "test_size": config.split.test_size,
                "stratify": config.split.stratify_by_target_quantiles,
            },
        )
        features, target = state.xy
        split_config = config.split

        stratify_labels = self._build_stratify_labels(target, split_config.stratify_by_target_quantiles)

        target_series = target if isinstance(target, pd.Series) else pd.Series(target)
        total_samples = len(target_series)
        if total_samples < _MIN_STRATIFY_CLASSES:
            message = "Need at least two samples to perform a train/test split."
            raise RuntimeError(message)

        test_count = max(1, min(total_samples - 1, math.ceil(total_samples * split_config.test_size)))
        rng = np.random.default_rng(split_config.random_state)

        labels_series: pd.Series | None = None
        if stratify_labels is not None:
            labels_series = stratify_labels.reindex(target_series.index)
            labels_series = None if labels_series.isna().any() else labels_series.reset_index(drop=True)

        if labels_series is not None:
            train_positions, test_positions = self._stratified_indices(labels_series, total_samples, test_count, rng)
        else:
            train_positions, test_positions = self._random_indices(total_samples, test_count, rng)

        x_train = self._select_rows(features, train_positions)
        x_test = self._select_rows(features, test_positions)
        y_train = self._select_rows(target, train_positions)
        y_test = self._select_rows(target, test_positions)

        state.split = cast("TrainTestSplitTuple", (x_train, x_test, y_train, y_test))
        state.add("split_done")
        state.logs.append("train_test_split")
        _LOGGER.info(
            "Train/test split complete.",
            extra={
                "event": "action_step",
                "action": self.schema.name,
                "stage": "train_test_split",
                "train_rows": len(y_train),
                "test_rows": len(y_test),
            },
        )

    @staticmethod
    def _build_stratify_labels(
        target: TargetVector | pd.Series,
        enabled: bool,
    ) -> pd.Series | None:
        """Return quantile labels for stratification when enabled and viable."""
        if not enabled:
            return None

        if not isinstance(target, pd.Series):
            target = pd.Series(target)

        unique_values = target.dropna().nunique()
        if unique_values < _MIN_STRATIFY_CLASSES:
            return None

        quantiles = min(_DEFAULT_QUANTILE_BINS, unique_values)
        if quantiles < _MIN_STRATIFY_CLASSES:
            return None
        quantile_positions = np.linspace(0.0, 1.0, num=int(quantiles) + 1)
        edges = np.asarray(target.quantile(quantile_positions), dtype=float)
        unique_edges = np.unique(edges)
        if unique_edges.size <= _MIN_STRATIFY_CLASSES:
            return None
        unique_edges[0] = float("-inf")
        unique_edges[-1] = float("inf")
        bins = np.digitize(np.asarray(target, dtype=float), unique_edges[1:-1], right=True)
        return pd.Series(bins, index=target.index)

    @staticmethod
    def _random_indices(
        total: int,
        test_count: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[int]]:
        indices = np.arange(total)
        rng.shuffle(indices)
        test_positions = indices[:test_count].tolist()
        train_positions = indices[test_count:].tolist()
        return train_positions, test_positions

    @staticmethod
    def _stratified_indices(
        labels: pd.Series,
        total: int,
        test_count: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[int]]:
        if len(labels) != total or test_count <= 0:
            return TrainTestSplit._random_indices(total, test_count, rng)

        grouped_list: dict[Any, list[int]] = {}
        for position, label in enumerate(labels):
            grouped_list.setdefault(label, []).append(position)

        if not grouped_list or len(grouped_list) > test_count:
            return TrainTestSplit._random_indices(total, test_count, rng)

        grouped_arrays = [np.array(indices, dtype=int) for indices in grouped_list.values()]
        group_sizes = np.array([len(indices) for indices in grouped_arrays], dtype=int)

        floor_counts = TrainTestSplit._compute_stratified_counts(group_sizes, test_count, total)
        if floor_counts is None:
            return TrainTestSplit._random_indices(total, test_count, rng)

        test_positions: list[int] = []
        train_positions: list[int] = []
        for count, indices in zip(floor_counts, grouped_arrays, strict=False):
            permuted = rng.permutation(indices)
            take = int(count)
            test_positions.extend(permuted[:take].tolist())
            train_positions.extend(permuted[take:].tolist())

        if not train_positions or not test_positions:
            return TrainTestSplit._random_indices(total, test_count, rng)

        train_array = rng.permutation(np.array(train_positions, dtype=int))
        test_array = rng.permutation(np.array(test_positions, dtype=int))
        return train_array.tolist(), test_array.tolist()

    @staticmethod
    def _compute_stratified_counts(
        group_sizes: np.ndarray,
        test_count: int,
        total: int,
    ) -> NDArray[np.int_] | None:
        test_ratio = test_count / total
        raw_counts = group_sizes * test_ratio
        floor_counts = np.minimum(group_sizes, np.floor(raw_counts).astype(int)).astype(int)
        remainder = test_count - int(floor_counts.sum())

        if remainder <= 0:
            return cast("NDArray[np.int_]", floor_counts)

        fractions = raw_counts - floor_counts
        for index in np.argsort(-fractions):
            if remainder <= 0:
                break
            available = group_sizes[index] - floor_counts[index]
            if available <= 0:
                continue
            take = min(available, remainder)
            floor_counts[index] += take
            remainder -= take

        if remainder > 0:
            return None

        return cast("NDArray[np.int_]", floor_counts)

    @staticmethod
    def _select_rows(data: Any, positions: Sequence[int]) -> Any:
        if hasattr(data, "iloc"):
            return data.iloc[list(positions)]
        if isinstance(data, np.ndarray):
            return cast("Any", data[positions])
        if hasattr(data, "take"):
            return data.take(positions, axis=0)
        return [data[pos] for pos in positions]
