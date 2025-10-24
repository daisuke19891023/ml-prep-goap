"""Tests for the mutable world state container."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goapml.models import WorldState


def test_world_state_add_has_remove() -> None:
    """Facts can be added, queried, and removed."""
    state = WorldState(facts={"file_exists"})

    assert state.has("file_exists")

    state.add("data_loaded")
    assert state.has("data_loaded")

    state.remove("file_exists")
    assert not state.has("file_exists")


def test_world_state_stores_runtime_artifacts() -> None:
    """Arbitrary runtime artefacts are stored on the state."""
    frame = pd.DataFrame({"feature": [1.0]})
    predictions = np.array([0.5], dtype=float)

    state = WorldState()
    state.df = frame
    state.encoding = "utf-8"
    state.target = "target"
    state.pred = predictions
    state.metrics = {"rmse": 0.1}
    state.logs.append("loaded")

    assert state.df is frame
    assert state.encoding == "utf-8"
    assert state.target == "target"
    assert state.pred is predictions
    assert state.metrics == {"rmse": 0.1}
    assert state.logs == ["loaded"]
