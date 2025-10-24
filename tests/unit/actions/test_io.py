"""Tests for I/O related actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from goapml.actions.io import DetectEncoding, LoadCSV
from goapml.models import FileSpec, PipelineConfig, WorldState

if TYPE_CHECKING:
    from pathlib import Path


def _build_config(path: str, encoding: str | None) -> PipelineConfig:
    file_spec = FileSpec(path=path, encoding=encoding, delimiter=",", decimal=".")
    return PipelineConfig(file=file_spec)


def test_detect_encoding_prefers_explicit_config(tmp_path: Path) -> None:
    """DetectEncoding should honour explicitly configured encodings."""
    csv_path = tmp_path / "data_utf8.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    config = _build_config(str(csv_path), "utf-8")
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding == "utf-8"
    assert state.has("encoding_detected")
    assert "encoding=utf-8" in state.logs


def test_detect_encoding_uses_chardet_when_unspecified(tmp_path: Path) -> None:
    """DetectEncoding should fall back to chardet when encoding is missing."""
    csv_path = tmp_path / "data_shift_jis.csv"
    csv_path.write_text("あ,い\nう,え\n", encoding="shift_jis")

    config = _build_config(str(csv_path), None)
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding == "shift_jis"
    assert state.has("encoding_detected")
    assert state.logs[-1] == "encoding=shift_jis"


def test_load_csv_reads_dataframe_into_state(tmp_path: Path) -> None:
    """LoadCSV should populate the world state's DataFrame and log its shape."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    config = _build_config(str(csv_path), None)
    state = WorldState(facts={"encoding_detected"}, encoding="utf-8")

    LoadCSV().run(state, config)

    assert state.df is not None
    assert tuple(state.df.columns) == ("a", "b")
    assert state.df.shape == (2, 2)
    assert state.has("csv_loaded")
    assert state.logs[-1] == "csv_shape=2x2"
