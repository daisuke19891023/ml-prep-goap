"""Tests for I/O related actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

from goapml.actions.io import (
    DetectEncoding,
    LoadCSV,
    MAX_SHIFT_JIS_SAMPLE_CHARS,
)
from goapml.models import FileSpec, PipelineConfig, WorldState

if TYPE_CHECKING:
    from pathlib import Path
    import pytest


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


def test_detect_encoding_limits_shift_jis_sample_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DetectEncoding should only read a bounded sample when probing Shift-JIS."""
    csv_path = tmp_path / "large_shift_jis.csv"
    csv_path.write_text("あ,い\nう,え\n", encoding="shift_jis")

    path_type = type(csv_path)
    original_open = path_type.open

    class DummyStream:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def read(self, size: int) -> str:
            assert size == MAX_SHIFT_JIS_SAMPLE_CHARS
            return "あいえお"

    def fake_open(self: Path, *args: Any, **kwargs: Any) -> object:
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == csv_path and mode == "r":
            encoding = kwargs.get("encoding")
            assert encoding in {"shift_jis", "cp932"}
            return DummyStream()
        return cast("object", original_open(self, *args, **kwargs))

    monkeypatch.setattr(path_type, "open", fake_open, raising=False)

    config = _build_config(str(csv_path), None)
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding == "shift_jis"


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
