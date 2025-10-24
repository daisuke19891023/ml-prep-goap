"""Identify the regression target column from a loaded CSV dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:
    from pandas import DataFrame

    from goapml.models import PipelineConfig, TargetSpec, WorldState

__all__ = ["IdentifyTarget"]


_TARGET_PRIORITY = ("target", "y", "label")


IDENTIFY_TARGET_SCHEMA = ActionSchema(
    name="identify_target",
    requires={"csv_loaded"},
    provides={"target_identified"},
    cost=1.0,
)


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
