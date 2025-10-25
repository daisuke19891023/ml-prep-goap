"""Pydantic schemas describing GOAP actions and goals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .models import PipelineConfig, WorldState


class ActionSchema(BaseModel):
    """Declarative description of an action's contract."""

    name: str = Field(..., description="Unique identifier for the action")
    requires: set[str] = Field(default_factory=set)
    provides: set[str] = Field(default_factory=set)
    cost: float = 1.0

    def is_applicable(self, state: WorldState) -> bool:
        """Return whether all preconditions are satisfied by the world state."""
        return self.requires.issubset(state.facts)

    def apply_effects(self, state: WorldState) -> None:
        """Record the facts that become true once the action succeeds."""
        for fact in self.provides:
            state.add(fact)


@dataclass(slots=True)
class Action:
    """Base implementation for concrete GOAP actions."""

    schema: ActionSchema

    def run(self, state: WorldState, config: PipelineConfig) -> None:  # pragma: no cover - interface
        """Execute the action, mutating the world state in-place."""
        msg = "Concrete actions must override run()."
        raise NotImplementedError(msg)


class Goal(BaseModel):
    """Describe the desired facts that define success."""

    required: set[str] = Field(default_factory=set)

    def is_satisfied(self, state: WorldState) -> bool:
        """Return whether the world state satisfies the goal."""
        return self.required.issubset(state.facts)


# Canonical action schema declarations used across the pipeline.

DETECT_ENCODING_SCHEMA = ActionSchema(
    name="detect_encoding",
    requires={"file_exists"},
    provides={"encoding_detected"},
    cost=1.0,
)


LOAD_CSV_SCHEMA = ActionSchema(
    name="load_csv",
    requires={"encoding_detected"},
    provides={"csv_loaded"},
    cost=1.0,
)


IDENTIFY_TARGET_SCHEMA = ActionSchema(
    name="identify_target",
    requires={"csv_loaded"},
    provides={"target_identified"},
    cost=1.0,
)


VALIDATE_TARGET_NUMERIC_SCHEMA = ActionSchema(
    name="validate_target_numeric",
    requires={"target_identified"},
    provides={"target_is_numeric"},
    cost=1.0,
)


SPLIT_XY_SCHEMA = ActionSchema(
    name="split_xy",
    requires={"target_is_numeric"},
    provides={"xy_separated"},
    cost=1.0,
)


TRAIN_TEST_SPLIT_SCHEMA = ActionSchema(
    name="train_test_split",
    requires={"xy_separated"},
    provides={"split_done"},
    cost=1.0,
)


CHECK_MISSING_SCHEMA = ActionSchema(
    name="check_missing",
    requires={"xy_separated"},
    provides={"missing_checked"},
    cost=1.0,
)


BUILD_PREPROCESSOR_SCHEMA = ActionSchema(
    name="build_preprocessor",
    requires={"split_done"},
    provides={"preprocessor_built"},
    cost=1.0,
)


FIT_TRANSFORM_PREPROCESSOR_SCHEMA = ActionSchema(
    name="fit_transform_preprocessor",
    requires={"preprocessor_built"},
    provides={"features_ready"},
    cost=1.0,
)


TRAIN_MODEL_SCHEMA = ActionSchema(
    name="train_model",
    requires={"features_ready"},
    provides={"trained"},
    cost=1.0,
)


PREDICT_SCHEMA = ActionSchema(
    name="predict",
    requires={"trained"},
    provides={"predicted"},
    cost=1.0,
)


EVALUATE_SCHEMA = ActionSchema(
    name="evaluate",
    requires={"predicted"},
    provides={"evaluated"},
    cost=1.0,
)


PERSIST_ARTIFACTS_SCHEMA = ActionSchema(
    name="persist_artifacts",
    requires={"trained"},
    provides={"persisted"},
    cost=1.0,
)


STANDARD_ACTION_SCHEMAS: tuple[ActionSchema, ...] = (
    DETECT_ENCODING_SCHEMA,
    LOAD_CSV_SCHEMA,
    IDENTIFY_TARGET_SCHEMA,
    VALIDATE_TARGET_NUMERIC_SCHEMA,
    SPLIT_XY_SCHEMA,
    TRAIN_TEST_SPLIT_SCHEMA,
    CHECK_MISSING_SCHEMA,
    BUILD_PREPROCESSOR_SCHEMA,
    FIT_TRANSFORM_PREPROCESSOR_SCHEMA,
    TRAIN_MODEL_SCHEMA,
    PERSIST_ARTIFACTS_SCHEMA,
    PREDICT_SCHEMA,
    EVALUATE_SCHEMA,
)
