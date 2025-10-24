"""Registry binding declarative action schemas to their implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from goapml.actions.evaluate import Evaluate
from goapml.actions.io import DetectEncoding, LoadCSV
from goapml.actions.preprocess import (
    BuildPreprocessor,
    CheckMissing,
    FitTransformPreprocessor,
)
from goapml.actions.split import SplitXY, TrainTestSplit
from goapml.actions.target import IdentifyTarget, ValidateTargetNumeric
from goapml.actions.train import Predict, TrainModel

import goapml.schemas as action_schemas
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from goapml.schemas import Action, ActionSchema
else:  # pragma: no cover - runtime aliases
    Action = action_schemas.Action
    ActionSchema = action_schemas.ActionSchema

__all__ = [
    "DuplicateActionError",
    "RegistryError",
    "UnknownActionError",
    "get",
    "list_schemas",
    "register",
]


class RegistryError(RuntimeError):
    """Base exception for registry related failures."""


class DuplicateActionError(RegistryError):
    """Raised when attempting to register an action whose name already exists."""

    def __init__(self, name: str) -> None:
        """Initialise the error with the conflicting action name."""
        super().__init__(f"Action '{name}' is already registered.")


class UnknownActionError(RegistryError):
    """Raised when requesting an action that is not present in the registry."""

    def __init__(self, name: str) -> None:
        """Initialise the error with the missing action name."""
        super().__init__(f"Action '{name}' is not registered.")


class RegisteredAction(BaseModel):
    """Pydantic representation of a registered action."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_schema: ActionSchema
    implementation: Action

    @property
    def name(self) -> str:
        """Return the unique name of the registered action."""
        return self.action_schema.name


class _ActionRegistry:
    """In-memory storage mapping schema names to concrete actions."""

    def __init__(self) -> None:
        self._actions: dict[str, RegisteredAction] = {}

    def register(self, action: Action) -> None:
        """Store the action, ensuring no duplicate schema names exist."""
        record = self._build_record(action)
        self._ensure_unique(record.name)
        self._actions[record.name] = record

    def get(self, name: str) -> Action:
        """Return the action implementation bound to ``name``."""
        record = self._actions.get(name)
        if record is None:
            raise UnknownActionError(name)
        return record.implementation

    def list_schemas(self) -> list[ActionSchema]:
        """Return all registered schemas preserving insertion order."""
        return [record.action_schema for record in self._actions.values()]

    def clear(self) -> None:
        """Remove every registered action (primarily for tests)."""
        self._actions.clear()

    def _build_record(self, action: Action) -> RegisteredAction:
        """Create a ``RegisteredAction`` validating the action's schema."""
        self._validate_schema(action.schema)
        return RegisteredAction(action_schema=action.schema, implementation=action)

    @staticmethod
    def _validate_schema(schema: ActionSchema) -> None:
        """Ensure the provided schema has a non-empty name."""
        if not schema.name:
            message = "Action schema must define a non-empty name."
            raise ValueError(message)

    def _ensure_unique(self, name: str) -> None:
        """Verify that the action name is not already present."""
        if name in self._actions:
            raise DuplicateActionError(name)


_REGISTRY = _ActionRegistry()


def _register_default_actions() -> None:
    """Populate the registry with the built-in action implementations."""
    default_actions = [
        DetectEncoding(),
        LoadCSV(),
        IdentifyTarget(),
        ValidateTargetNumeric(),
        SplitXY(),
        TrainTestSplit(),
        CheckMissing(),
        BuildPreprocessor(),
        FitTransformPreprocessor(),
        TrainModel(),
        Predict(),
        Evaluate(),
    ]

    for action in default_actions:
        _REGISTRY.register(action)


def register(action: Action) -> None:
    """Register an action implementation with its schema."""
    _REGISTRY.register(action)


def get(name: str) -> Action:
    """Retrieve an action implementation by its schema name."""
    return _REGISTRY.get(name)


def list_schemas() -> list[ActionSchema]:
    """Return every registered schema."""
    return _REGISTRY.list_schemas()


_register_default_actions()
