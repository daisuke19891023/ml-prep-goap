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
