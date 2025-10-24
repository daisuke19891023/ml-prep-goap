"""Identify the regression target column from a loaded CSV dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import math
import numbers
import statistics

from goapml.schemas import (
    Action,
    ActionSchema,
    IDENTIFY_TARGET_SCHEMA,
    VALIDATE_TARGET_NUMERIC_SCHEMA,
)

if TYPE_CHECKING:
    from pandas import DataFrame

    from goapml.models import PipelineConfig, TargetSpec, WorldState

__all__ = ["IdentifyTarget", "ValidateTargetNumeric"]


_TARGET_PRIORITY = ("target", "y", "label")


def _coerce_to_float(value: object) -> float:
    """Attempt to convert ``value`` to ``float`` returning NaN on failure."""
    if isinstance(value, numbers.Real):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return math.nan
        try:
            return float(text)
        except ValueError:
            return math.nan

    return math.nan


@dataclass(slots=True)
class IdentifyTarget(Action):
    """Determine the name of the target column according to the configured strategy."""

    schema: ActionSchema = field(default_factory=lambda: IDENTIFY_TARGET_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Select the target column and store it on the world state."""
        if state.df is None:
            message = "CSV must be loaded before identifying the target column."
            raise RuntimeError(message)

        target_name = self._resolve_target(state.df, config.target)

        state.target = target_name
        state.add("target_identified")
        state.logs.append(f"target:{target_name}")

    def _resolve_target(self, df: DataFrame, spec: TargetSpec) -> str:
        """Dispatch to the configured target identification strategy."""
        if spec.strategy == "explicit":
            return self._select_explicit(df, spec)
        if spec.strategy == "by_name_heuristic":
            return self._select_by_name(df)
        if spec.strategy == "last_column":
            return self._select_last_column(df)

        message = f"Unknown target identification strategy: {spec.strategy}"
        raise ValueError(message)

    @staticmethod
    def _select_explicit(df: DataFrame, spec: TargetSpec) -> str:
        """Return the explicitly configured target column."""
        if not spec.name:
            message = "Target name must be provided when using the explicit strategy."
            raise ValueError(message)

        if spec.name not in df.columns:
            message = f"Target column '{spec.name}' not found in the dataset."
            raise ValueError(message)

        return spec.name

    @staticmethod
    def _select_by_name(df: DataFrame) -> str:
        """Choose a target column based on name heuristics."""
        columns = list(map(str, df.columns))
        lower_map = {column: column.strip().lower() for column in columns}

        for candidate in _TARGET_PRIORITY:
            for column, lowered in lower_map.items():
                if lowered == candidate:
                    return column

        for column, lowered in lower_map.items():
            if lowered.endswith("_y"):
                return column

        message = "No suitable target column found via heuristics."
        raise ValueError(message)

    @staticmethod
    def _select_last_column(df: DataFrame) -> str:
        """Return the final column name of the dataset."""
        if len(df.columns) == 0:
            message = "Cannot select target column from empty DataFrame."
            raise ValueError(message)

        raw_columns = cast("list[object]", df.columns.tolist())
        columns: list[str] = [str(column) for column in raw_columns]
        return columns[-1]


@dataclass(slots=True)
class ValidateTargetNumeric(Action):
    """Ensure the identified target column can be treated as numeric."""

    schema: ActionSchema = field(default_factory=lambda: VALIDATE_TARGET_NUMERIC_SCHEMA)
    nan_ratio_threshold: float = 0.2

    def run(self, state: WorldState, config: PipelineConfig) -> None:  # noqa: ARG002
        """Convert the target column to numeric data, imputing sparse failures."""
        if state.df is None or state.target is None:
            message = "Target must be identified before numeric validation."
            raise RuntimeError(message)

        series = state.df[state.target]
        numeric_values = [_coerce_to_float(value) for value in series]
        total = len(numeric_values)
        if total == 0:
            message = "Target column contains no rows to validate."
            raise ValueError(message)

        nan_count = sum(math.isnan(value) for value in numeric_values)
        nan_ratio = nan_count / total

        if nan_ratio > self.nan_ratio_threshold:
            state.logs.append(
                f"target_numeric_failed:nan_ratio={nan_ratio:.3f}",
            )
            percentage = nan_ratio * 100
            message = (
                "Target column could not be coerced to numeric values: "
                f"{percentage:.1f}% NaNs after conversion"
            )
            raise ValueError(message)

        non_nan_values = [value for value in numeric_values if not math.isnan(value)]
        median = float(statistics.median(non_nan_values))
        filled_values = [median if math.isnan(value) else value for value in numeric_values]
        state.df[state.target] = filled_values
        state.add("target_is_numeric")
        state.logs.append(
            f"target_numeric_ok:nan_ratio={nan_ratio:.3f},imputed=median",
        )
