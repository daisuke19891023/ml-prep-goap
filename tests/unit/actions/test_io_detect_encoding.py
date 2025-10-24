"""Tests for I/O DetectEncoding action.

Covers explicit encoding, chardet-based detection (UTF-8 and Shift-JIS), and fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

from goapml.actions import DetectEncoding
from goapml.models import FileSpec, PipelineConfig, WorldState


def _make_config(csv_path: Path, encoding: str | None) -> PipelineConfig:
    """Construct a minimal pipeline config for a given file."""
    return PipelineConfig(
        file=FileSpec(
            path=str(csv_path),
            encoding=encoding,
            delimiter=",",
            decimal=".",
            has_header=True,
        ),
    )


def test_detect_encoding_uses_explicit_and_logs(tmp_path: Path) -> None:
    """When explicit encoding is provided, it is used and logged."""
    csv_path = tmp_path / "explicit.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    config = _make_config(csv_path, encoding="utf-8")
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding == "utf-8"
    assert state.has("encoding_detected")
    assert any(entry.startswith("encoding=utf-8") and "source=explicit" in entry for entry in state.logs)


def test_detect_encoding_detects_utf8_and_logs(tmp_path: Path) -> None:
    """Detect UTF-8 encoding when not explicitly specified."""
    csv_path = tmp_path / "utf8.csv"
    # Include non-ASCII to avoid ASCII detection
    csv_path.write_text("特徴,目標\n値,1\n", encoding="utf-8")

    config = _make_config(csv_path, encoding=None)
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding is not None
    assert state.encoding.lower() == "utf-8"
    assert state.has("encoding_detected")
    assert any(entry.startswith("encoding=") for entry in state.logs)


def test_detect_encoding_sets_encoding_for_shift_jis_and_logs(tmp_path: Path) -> None:
    """Set some encoding for Shift-JIS CSV and log it (label may vary)."""
    csv_path = tmp_path / "sjis.csv"
    # Japanese content encoded in CP932 (Windows-31J, Shift_JIS superset)
    content = "特徴,目標\n漢字仮名交じり文,1\n".encode("cp932")
    csv_path.write_bytes(content)

    config = _make_config(csv_path, encoding=None)
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding is not None
    assert state.has("encoding_detected")
    assert any(entry.startswith("encoding=") for entry in state.logs)


def test_detect_encoding_fallbacks_to_utf8_on_empty_file(tmp_path: Path) -> None:
    """Fallback to UTF-8 when detection yields no encoding (e.g., empty file)."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_bytes(b"")

    config = _make_config(csv_path, encoding=None)
    state = WorldState(facts={"file_exists"})

    DetectEncoding().run(state, config)

    assert state.encoding == "utf-8"
    assert state.has("encoding_detected")
    assert any(entry.startswith("encoding=utf-8") for entry in state.logs)
