from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from goapml.engine import ActionExecutionError, execute_with_replanning
from goapml.models import FileSpec, PipelineConfig, TargetSpec, WorldState
from goapml.schemas import Goal


if TYPE_CHECKING:
    from pathlib import Path


def _build_config(tmp_path: Path, target_name: str | None) -> PipelineConfig:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text(
        "num,cat,target\n"
        "1.0,a,10.0\n"
        "2.0,b,20.0\n"
        "3.0,a,30.0\n"
        "4.0,b,40.0\n",
        encoding="utf-8",
    )
    file_spec = FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )
    target_spec = TargetSpec(strategy="explicit", name=target_name)
    return PipelineConfig(file=file_spec, target=target_spec)


def test_execute_with_replanning_reaches_goal(tmp_path: Path) -> None:
    """The engine should reach the evaluated goal under normal conditions."""
    config = _build_config(tmp_path, target_name="target")
    state = WorldState(facts={"file_exists"})
    goal = Goal(required={"evaluated"})

    executed = execute_with_replanning(state=state, config=config, goal=goal)

    assert goal.is_satisfied(state)
    assert state.metrics is not None
    assert executed[-1] == "evaluate"
    assert any(log.startswith("evaluate:") for log in state.logs)


def test_execute_with_replanning_stops_after_repeated_failure(tmp_path: Path) -> None:
    """Replanning should eventually raise when an action keeps failing."""
    config = _build_config(tmp_path, target_name="missing")
    state = WorldState(facts={"file_exists"})
    goal = Goal(required={"evaluated"})

    with pytest.raises(ActionExecutionError) as exc_info:
        execute_with_replanning(state=state, config=config, goal=goal)

    error = exc_info.value
    assert error.action_name == "identify_target"
    assert not goal.is_satisfied(state)

    failure_logs = [log for log in state.logs if log.startswith("action_error:identify_target")]
    assert len(failure_logs) >= 2
