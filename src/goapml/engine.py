"""Execution engine orchestrating GOAP planning with replanning support."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from goapml.actions import registry
from goapml.planner import plan

if TYPE_CHECKING:
    from goapml.models import PipelineConfig, WorldState
    from goapml.schemas import ActionSchema, Goal

__all__ = [
    "ActionExecutionError",
    "ExecutionError",
    "PlanningError",
    "execute_with_replanning",
]


_LOGGER = logging.getLogger(__name__)
_MAX_CONSECUTIVE_FAILURES = 2


class ExecutionError(RuntimeError):
    """Base exception raised when executing a GOAP plan fails."""


class PlanningError(ExecutionError):
    """Raised when the planner cannot produce a valid action sequence."""


class ActionExecutionError(ExecutionError):
    """Raised when an action continues to fail after replanning attempts."""

    def __init__(self, action_name: str, cause: Exception) -> None:
        """Store the failing action and original error."""
        message = f"Action '{action_name}' failed repeatedly: {cause}"
        super().__init__(message)
        self.action_name = action_name
        self.cause = cause


def execute_with_replanning(
    *,
    state: WorldState,
    config: PipelineConfig,
    goal: Goal,
    actions: list[ActionSchema] | None = None,
) -> list[str]:
    """Execute a plan, replanning around recoverable failures.

    Returns the ordered list of action names that were executed successfully.
    """
    base_actions = registry.list_schemas() if actions is None else list(actions)

    executed: list[str] = []
    failure_counts: dict[str, int] = defaultdict(int)
    blocked_actions: set[str] = set()
    last_failure: tuple[str, Exception] | None = None

    _LOGGER.info(
        "Starting GOAP execution.",
        extra={
            "event": "engine_start",
            "available_actions": [schema.name for schema in base_actions],
            "goal": sorted(goal.required),
        },
    )

    while not goal.is_satisfied(state):
        plan_actions = _plan_next_actions(
            base_actions,
            state,
            goal,
            config,
            blocked_actions,
            last_failure,
        )

        if plan_actions is None:
            _LOGGER.info(
                "Replanning requested after resolving blocked actions.",
                extra={
                    "event": "plan_retry",
                    "blocked": sorted(blocked_actions),
                },
            )
            continue

        plan_failed, last_failure = _execute_plan(
            plan_actions,
            state,
            config,
            goal,
            executed,
            failure_counts,
            blocked_actions,
        )

        if plan_failed:
            _LOGGER.info(
                "Plan execution failed; attempting to replan.",
                extra={
                    "event": "plan_retry",
                    "blocked": sorted(blocked_actions),
                },
            )
            continue

        blocked_actions.clear()

    _LOGGER.info(
        "Execution completed successfully.",
        extra={
            "event": "engine_complete",
            "executed_actions": list(executed),
        },
    )

    return executed


def _filter_actions(
    actions: list[ActionSchema],
    blocked: set[str],
) -> list[ActionSchema]:
    """Return the subset of actions not currently blocked."""
    if not blocked:
        return list(actions)
    return [schema for schema in actions if schema.name not in blocked]


def _apply_effects(state: WorldState, schema: ActionSchema) -> None:
    """Record the schema's provided facts on the world state."""
    for fact in schema.provides:
        state.add(fact)


def _plan_next_actions(
    base_actions: list[ActionSchema],
    state: WorldState,
    goal: Goal,
    config: PipelineConfig,
    blocked_actions: set[str],
    last_failure: tuple[str, Exception] | None,
) -> list[ActionSchema] | None:
    """Return the next plan or ``None`` when a replan should be attempted."""
    available_actions = _filter_actions(base_actions, blocked_actions)
    if not available_actions:
        _LOGGER.error(
            "No actions available to satisfy the goal.",
            extra={
                "event": "planning_failed",
                "reason": "no_actions",
            },
        )
        _raise_planning_error(last_failure, "No actions available to satisfy the goal.")

    plan_actions = plan(
        initial_facts=state.facts,
        goal_facts=goal.required,
        actions=available_actions,
        max_expansions=config.planner.max_expansions,
    )

    if plan_actions:
        _LOGGER.info(
            "Plan generated successfully.",
            extra={
                "event": "plan_generated",
                "actions": [schema.name for schema in plan_actions],
            },
        )
        return plan_actions

    if blocked_actions:
        _LOGGER.info(
            "Clearing blocked actions before replanning.",
            extra={
                "event": "plan_blocked",
                "blocked": sorted(blocked_actions),
            },
        )
        blocked_actions.clear()
        return None

    _LOGGER.error(
        "Unable to produce a plan to reach the goal.",
        extra={
            "event": "planning_failed",
            "reason": "no_plan",
        },
    )
    _raise_planning_error(last_failure, "Unable to produce a plan to reach the goal.")
    return None


def _execute_plan(
    plan_actions: list[ActionSchema],
    state: WorldState,
    config: PipelineConfig,
    goal: Goal,
    executed: list[str],
    failure_counts: dict[str, int],
    blocked_actions: set[str],
) -> tuple[bool, tuple[str, Exception] | None]:
    """Execute a plan returning a flag indicating whether replanning is required."""
    last_failure: tuple[str, Exception] | None = None

    for schema in plan_actions:
        action = registry.get(schema.name)
        _LOGGER.info(
            "Executing action.",
            extra={"event": "action_start", "action": schema.name},
        )
        try:
            action.run(state, config)
        except Exception as exc:  # pragma: no cover - broad to support replanning
            _LOGGER.exception(
                "Action '%s' failed during execution.",
                schema.name,
                extra={"event": "action_failure", "action": schema.name},
            )
            state.logs.append(
                f"action_error:{schema.name}:{exc.__class__.__name__}:{exc}",
            )
            failure_counts[schema.name] += 1
            last_failure = (schema.name, exc)
            if failure_counts[schema.name] >= _MAX_CONSECUTIVE_FAILURES:
                _LOGGER.error(
                    "Action '%s' exceeded maximum retries.",
                    schema.name,
                    extra={
                        "event": "action_aborted",
                        "action": schema.name,
                        "failures": failure_counts[schema.name],
                    },
                )
                raise ActionExecutionError(schema.name, exc) from exc
            blocked_actions.add(schema.name)
            return True, last_failure

        failure_counts[schema.name] = 0
        blocked_actions.discard(schema.name)
        last_failure = None
        _apply_effects(state, schema)
        executed.append(schema.name)
        _LOGGER.info(
            "Action completed successfully.",
            extra={"event": "action_complete", "action": schema.name},
        )

        if goal.is_satisfied(state):
            return False, None

    return False, last_failure


def _raise_planning_error(
    last_failure: tuple[str, Exception] | None,
    message: str,
) -> None:
    """Raise an appropriate planning error based on the recorded failure."""
    if last_failure is not None:
        name, error = last_failure
        _LOGGER.error(
            "Raising action execution error due to persistent failure.",
            extra={
                "event": "planning_failed",
                "action": name,
                "reason": "action_failure",
            },
        )
        raise ActionExecutionError(name, error) from error

    error_message = message
    _LOGGER.error(
        "Raising planning error.",
        extra={
            "event": "planning_failed",
            "reason": "planning_error",
        },
    )
    raise PlanningError(error_message)
