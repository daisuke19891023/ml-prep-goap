"""Unit tests for action schemas and goals."""

from __future__ import annotations

from goapml.models import WorldState
from goapml.schemas import ActionSchema, Goal


def test_action_schema_precondition_check_and_effect_application() -> None:
    """Requires facts must be present and provides facts are recorded."""
    state = WorldState(facts={"file_loaded"})
    schema = ActionSchema(
        name="clean_data",
        requires={"file_loaded"},
        provides={"data_clean"},
        cost=2.0,
    )

    assert schema.is_applicable(state)

    schema.apply_effects(state)

    assert state.has("data_clean")


def test_action_schema_precondition_failure() -> None:
    """Preconditions that are not satisfied cause the schema to fail the check."""
    state = WorldState(facts={"file_loaded"})
    schema = ActionSchema(name="train", requires={"features_ready"})

    assert not schema.is_applicable(state)


def test_goal_satisfaction() -> None:
    """Goals report satisfaction when all required facts are present."""
    goal = Goal(required={"evaluated"})
    state = WorldState(facts={"evaluated"})

    assert goal.is_satisfied(state)

    state.remove("evaluated")

    assert not goal.is_satisfied(state)
