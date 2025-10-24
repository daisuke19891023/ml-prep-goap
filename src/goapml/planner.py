"""Planning utilities for GOAP-based pipelines."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .schemas import ActionSchema
else:  # pragma: no cover - runtime fallback for annotations
    ActionSchema = Any


class PlannerState(BaseModel):
    """Immutable wrapper for the planner's fact set."""

    model_config = ConfigDict(frozen=True)

    facts: frozenset[str]

    @classmethod
    def from_iterable(cls, facts: Iterable[str]) -> PlannerState:
        """Construct a planner state from the given facts."""
        return cls(facts=frozenset(facts))

    def apply(self, action: ActionSchema) -> PlannerState:
        """Return the successor state after applying the action's effects."""
        return PlannerState(facts=frozenset(self.facts.union(action.provides)))

    def satisfies(self, goal_facts: set[str]) -> bool:
        """Return whether the goal facts are a subset of the state's facts."""
        return goal_facts.issubset(self.facts)

    def __hash__(self) -> int:
        """Allow planner states to be used as dictionary keys."""
        return hash(self.facts)


class GoalDefinition(BaseModel):
    """Goal specification with helper heuristics."""

    model_config = ConfigDict(frozen=True)

    facts: frozenset[str]

    @classmethod
    def from_iterable(cls, facts: Iterable[str]) -> GoalDefinition:
        """Construct a goal definition from the provided fact collection."""
        return cls(facts=frozenset(facts))

    def is_satisfied(self, state: PlannerState) -> bool:
        """Return whether the goal is satisfied for the given state."""
        return self.facts.issubset(state.facts)

    def heuristic(self, state: PlannerState) -> float:
        """Return the admissible heuristic value for a state."""
        return float(len(self.facts.difference(state.facts)))


class TransitionRecord(BaseModel):
    """Record describing how a planner state was reached."""

    previous_state: PlannerState
    action: ActionSchema


class FrontierEntry(BaseModel):
    """Item stored in the frontier priority queue."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    priority: float
    order: int
    state: PlannerState

    def __lt__(self, other: object) -> bool:
        """Order frontier entries by priority, then insertion order."""
        if not isinstance(other, FrontierEntry):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.order < other.order


def _frontier_factory() -> list[FrontierEntry]:
    """Return a new empty frontier list."""
    return []


def _score_factory() -> dict[PlannerState, float]:
    """Return a fresh score mapping."""
    return {}


def _transition_factory() -> dict[PlannerState, TransitionRecord]:
    """Return a fresh transition mapping."""
    return {}


class SearchContext(BaseModel):
    """Mutable structures required to perform A* search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frontier: list[FrontierEntry] = Field(default_factory=_frontier_factory)
    g_score: dict[PlannerState, float] = Field(default_factory=_score_factory)
    came_from: dict[PlannerState, TransitionRecord] = Field(default_factory=_transition_factory)
    next_order: int = Field(default=0)

    def has_entries(self) -> bool:
        """Return whether the frontier still contains entries."""
        return bool(self.frontier)

    def push(self, *, state: PlannerState, priority: float) -> None:
        """Insert a new entry into the frontier with the given priority."""
        order = self.next_order
        self.next_order += 1
        heappush(
            self.frontier,
            FrontierEntry(priority=priority, order=order, state=state),
        )

    def pop(self) -> FrontierEntry:
        """Remove and return the next frontier entry."""
        return heappop(self.frontier)

    def record_transition(
        self,
        *,
        next_state: PlannerState,
        previous_state: PlannerState,
        action: ActionSchema,
        cost: float,
        goal: GoalDefinition,
    ) -> None:
        """Register the best known path to a successor state."""
        self.came_from[next_state] = TransitionRecord(
            previous_state=previous_state,
            action=action,
        )
        self.g_score[next_state] = cost
        self.push(state=next_state, priority=cost + goal.heuristic(next_state))


def _reconstruct_plan(
    came_from: Mapping[PlannerState, TransitionRecord],
    current: PlannerState,
    start: PlannerState,
) -> list[ActionSchema]:
    """Backtrack the plan from the goal state to the start state."""
    actions: list[ActionSchema] = []
    state = current
    while state != start:
        transition = came_from[state]
        actions.append(transition.action)
        state = transition.previous_state
    actions.reverse()
    return actions


class PlannerEngine(BaseModel):
    """Coordinator responsible for executing the A* planning algorithm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal: GoalDefinition
    actions: list[ActionSchema]
    max_expansions: int
    context: SearchContext

    def initialize(self, start_state: PlannerState) -> None:
        """Seed the search context with the starting state."""
        self.context.g_score[start_state] = 0.0
        self.context.push(state=start_state, priority=self.goal.heuristic(start_state))

    def execute(self, start_state: PlannerState) -> list[ActionSchema]:
        """Perform the A* search and return the resulting plan."""
        if self.goal.is_satisfied(start_state):
            return []

        self.initialize(start_state)
        expansions = 0

        while self.context.has_entries() and expansions < self.max_expansions:
            current_entry = self.context.pop()
            current_state = current_entry.state
            expansions += 1

            if self.goal.is_satisfied(current_state):
                return _reconstruct_plan(
                    self.context.came_from, current_state, start_state,
                )

            self._expand_from(current_state)

        return []

    def _expand_from(self, state: PlannerState) -> None:
        """Explore applicable actions from the provided state."""
        current_cost = self.context.g_score[state]
        for action in self.actions:
            if not action.requires.issubset(state.facts):
                continue

            next_state = state.apply(action)
            tentative_cost = current_cost + action.cost
            known_cost = self.context.g_score.get(next_state, float("inf"))

            if tentative_cost >= known_cost:
                continue

            self.context.record_transition(
                next_state=next_state,
                previous_state=state,
                action=action,
                cost=tentative_cost,
                goal=self.goal,
            )


def plan(
    *,
    initial_facts: Iterable[str],
    goal_facts: Iterable[str],
    actions: Iterable[ActionSchema],
    max_expansions: int = 10_000,
) -> list[ActionSchema]:
    """Run an A* search over fact sets to satisfy the provided goal."""
    start_state = PlannerState.from_iterable(initial_facts)
    goal = GoalDefinition.from_iterable(goal_facts)
    engine = PlannerEngine(
        goal=goal,
        actions=list(actions),
        max_expansions=max_expansions,
        context=SearchContext(),
    )
    return engine.execute(start_state)
