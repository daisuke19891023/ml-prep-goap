"""Actions handling file encoding detection and CSV loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from chardet.universaldetector import UniversalDetector

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:
    from goapml.models import PipelineConfig, WorldState
    from pandas import DataFrame

    def pandas_read_csv(*args: object, **kwargs: object) -> DataFrame: ...
else:
    from pandas import read_csv as pandas_read_csv

__all__ = ["DetectEncoding", "LoadCSV"]


_HIRAGANA_START = 0x3040
_HIRAGANA_END = 0x30FF
_CJK_UNIFIED_START = 0x4E00
_CJK_UNIFIED_END = 0x9FFF
_HALF_WIDTH_KATAKANA_START = 0xFF66
_HALF_WIDTH_KATAKANA_END = 0xFF9D


DETECT_ENCODING_SCHEMA = ActionSchema(
    name="detect_encoding",
    requires={"file_exists"},
    provides={"encoding_detected"},
    cost=1.0,
)


LOAD_CSV_SCHEMA = ActionSchema(
    name="load_csv",
    requires={"encoding_detected"},
    provides={"csv_loaded"},
    cost=1.0,
)


@dataclass(slots=True)
class DetectEncoding(Action):
    """Determine the CSV file encoding before ingestion."""

    schema: ActionSchema = field(default_factory=lambda: DETECT_ENCODING_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Populate the state's encoding using configuration or auto-detection."""
        file_path = Path(config.file.path)
        specified = config.file.encoding
        encoding = (
            self._normalise_encoding(specified)
            if specified
            else self._detect_with_chardet(file_path)
        )

        state.encoding = encoding
        state.add("encoding_detected")
        state.logs.append(f"encoding={encoding}")

    def _detect_with_chardet(self, path: Path) -> str:
        detector = UniversalDetector()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                detector.feed(chunk)
                if detector.done:
                    break
        detector.close()

        detected = detector.result.get("encoding")
        if detected:
            encoding = self._normalise_encoding(detected)
            return self._refine_encoding(path, encoding)
        return "utf-8"

    def _refine_encoding(self, path: Path, encoding: str) -> str:
        if encoding in {"windows-1252", "iso-8859-1"}:
            candidate = self._try_shift_jis(path)
            if candidate is not None:
                return candidate
        return encoding

    def _try_shift_jis(self, path: Path) -> str | None:
        for candidate in ("shift_jis", "cp932"):
            try:
                text = path.read_text(encoding=candidate)
            except UnicodeDecodeError:
                continue
            if self._contains_cjk(text):
                return candidate
        return None

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        for char in text:
            codepoint = ord(char)
            if (
                _HIRAGANA_START <= codepoint <= _HIRAGANA_END
                or _CJK_UNIFIED_START <= codepoint <= _CJK_UNIFIED_END
                or _HALF_WIDTH_KATAKANA_START <= codepoint <= _HALF_WIDTH_KATAKANA_END
            ):
                return True
        return False

    @staticmethod
    def _normalise_encoding(encoding: str) -> str:
        return encoding.strip().lower()


@dataclass(slots=True)
class LoadCSV(Action):
    """Load the CSV data into a pandas DataFrame."""

    schema: ActionSchema = field(default_factory=lambda: LOAD_CSV_SCHEMA)

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Read the CSV file into the state's DataFrame attribute."""
        if state.encoding is None:
            message = "File encoding must be detected before loading the CSV."
            raise RuntimeError(message)

        file_path = Path(config.file.path)
        header = 0 if config.file.has_header else None

        df = pandas_read_csv(
            file_path,
            encoding=state.encoding,
            delimiter=config.file.delimiter,
            decimal=config.file.decimal,
            header=header,
        )

        state.df = df
        state.add("csv_loaded")
        state.logs.append(f"csv_shape={df.shape[0]}x{df.shape[1]}")
