"""Unit tests for the GOAP A* planner."""

from __future__ import annotations

from goapml.planner import GoalDefinition, PlannerState, SearchContext, plan
from goapml.schemas import ActionSchema


def _make_actions() -> list[ActionSchema]:
    return [
        ActionSchema(
            name="load_data",
            requires={"csv_available"},
            provides={"data_loaded"},
            cost=1.0,
        ),
        ActionSchema(
            name="prepare_features",
            requires={"data_loaded"},
            provides={"features_ready"},
            cost=1.0,
        ),
        ActionSchema(
            name="train_model",
            requires={"features_ready"},
            provides={"model_trained"},
            cost=1.0,
        ),
    ]
def test_goal_definition_reports_progress() -> None:
    """Goal definition should report satisfaction and heuristic values."""
    state = PlannerState.from_iterable(["a"])
    goal = GoalDefinition.from_iterable(["a", "b"])

    assert not goal.is_satisfied(state)
    assert goal.heuristic(state) == 1.0

    progressed_state = PlannerState.from_iterable(["a", "b"])
    assert goal.is_satisfied(progressed_state)
    assert goal.heuristic(progressed_state) == 0.0


def test_search_context_orders_frontier_entries() -> None:
    """Search context should always pop the lowest-priority state first."""
    context = SearchContext()
    state_low = PlannerState.from_iterable(["low"])
    state_high = PlannerState.from_iterable(["high"])

    context.push(state=state_high, priority=5.0)
    context.push(state=state_low, priority=1.0)

    first_entry = context.pop()
    assert first_entry.state == state_low


def test_planner_returns_sequence_for_linear_dependencies() -> None:
    """The planner should chain actions that satisfy sequential dependencies."""
    actions = _make_actions()
    plan_result = plan(
        initial_facts={"csv_available"},
        goal_facts={"model_trained"},
        actions=actions,
    )

    assert [action.name for action in plan_result] == [
        "load_data",
        "prepare_features",
        "train_model",
    ]


def test_planner_shortens_plan_when_goal_partially_satisfied() -> None:
    """Existing facts should allow the planner to skip redundant actions."""
    actions = _make_actions()
    plan_result = plan(
        initial_facts={"csv_available", "data_loaded"},
        goal_facts={"model_trained"},
        actions=actions,
    )

    assert [action.name for action in plan_result] == [
        "prepare_features",
        "train_model",
    ]


def test_planner_returns_empty_plan_for_dead_end() -> None:
    """Unreachable goals must yield an empty plan rather than looping."""
    actions = [
        ActionSchema(
            name="loop",
            requires={"csv_available"},
            provides={"csv_available"},
            cost=1.0,
        ),
    ]

    plan_result = plan(
        initial_facts={"csv_available"},
        goal_facts={"model_trained"},
        actions=actions,
    )

    assert plan_result == []
