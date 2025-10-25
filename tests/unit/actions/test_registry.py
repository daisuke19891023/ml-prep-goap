"""Tests for the action registry bindings."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from goapml.actions import registry
from goapml.models import FileSpec, PipelineConfig, WorldState
from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def reset_registry() -> Iterator[None]:
    """Reload the registry module before and after each test."""
    importlib.reload(registry)
    yield
    importlib.reload(registry)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Create a minimal pipeline configuration backed by a temporary CSV."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("feature,target\n1,2\n")
    file_spec = FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )
    return PipelineConfig(file=file_spec)


class DummyAction(Action):
    """Simple action that records its execution on the world state."""

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Record execution results on the world state."""
        state.add("dummy_done")
        state.logs.append(f"ran:{self.schema.name}:{config.file.path}")


def test_register_get_and_execute_action(pipeline_config: PipelineConfig) -> None:
    """Action registration enables retrieval and execution through the registry."""
    dummy_schema = ActionSchema(
        name="dummy",
        requires=set(),
        provides={"dummy_done"},
        cost=1.0,
    )
    dummy_action = DummyAction(schema=dummy_schema)

    existing_schemas = registry.list_schemas()
    registry.register(dummy_action)
    retrieved = registry.get("dummy")

    state = WorldState()
    retrieved.run(state, pipeline_config)

    assert state.has("dummy_done")
    assert state.logs[-1].startswith("ran:dummy:")
    assert registry.list_schemas() == [*existing_schemas, dummy_schema]


def test_register_duplicate_name_raises() -> None:
    """Registering two actions with the same name raises an error."""
    schema = ActionSchema(name="duplicate", requires=set(), provides=set())
    registry.register(DummyAction(schema=schema))

    with pytest.raises(registry.DuplicateActionError):
        registry.register(DummyAction(schema=schema))


def test_get_unknown_action_raises() -> None:
    """Requesting an unknown action results in a dedicated exception."""
    with pytest.raises(registry.UnknownActionError):
        registry.get("missing")


def test_default_action_schemas_are_registered() -> None:
    """Importing the registry registers the standard set of schemas."""
    names = [schema.name for schema in registry.list_schemas()]
    assert names == DEFAULT_ACTION_NAMES
DEFAULT_ACTION_NAMES = [
    "detect_encoding",
    "load_csv",
    "identify_target",
    "validate_target_numeric",
    "split_xy",
    "train_test_split",
    "check_missing",
    "build_preprocessor",
    "fit_transform_preprocessor",
    "train_model",
    "persist_artifacts",
    "predict",
    "evaluate",
]


