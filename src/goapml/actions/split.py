"""Actions handling feature/target separation for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:

    from goapml.models import PipelineConfig, WorldState

__all__ = ["SplitXY"]


SPLIT_XY_SCHEMA = ActionSchema(
    name="split_xy",
    requires={"target_is_numeric"},
    provides={"xy_separated"},
    cost=1.0,
)


@dataclass(slots=True)
class SplitXY(Action):
    """Split the dataframe into features and target while inferring column types."""

    schema: ActionSchema = field(default_factory=lambda: SPLIT_XY_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:  # noqa: ARG002
        """Separate X/y and determine feature column categories."""
        if state.df is None or state.target is None:
            message = "Target must be validated as numeric before splitting."
            raise RuntimeError(message)

        features = state.df.drop(columns=[state.target])
        target = state.df[state.target]

        state.xy = (features, target)
        state.col_types = {
            column: "numeric" if is_numeric_dtype(features[column]) else "categorical"
            for column in features.columns
        }
        state.add("xy_separated")
        state.logs.append("split_xy")
