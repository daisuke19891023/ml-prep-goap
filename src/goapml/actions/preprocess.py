"""Actions responsible for preprocessing diagnostics and transformations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pandas as pd

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:
    from pandas import DataFrame

    from goapml.models import PipelineConfig, WorldState

__all__ = ["CheckMissing"]


_LOGGER = logging.getLogger(__name__)


_CHECK_MISSING_SCHEMA = ActionSchema(
    name="check_missing",
    requires={"xy_separated"},
    provides={"missing_checked"},
    cost=1.0,
)


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
