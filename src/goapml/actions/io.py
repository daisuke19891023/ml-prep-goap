"""I/O related action implementations.

T7. DetectEncoding
-------------------
Detect file encoding prior to CSV loading.

Behaviour:
- If an explicit encoding is provided in the pipeline config, use it.
- Otherwise attempt automatic detection using ``chardet``.
- On detection failure, fall back to ``"utf-8"``.

Schema:
- requires: {"file_exists"}
- provides: {"encoding_detected"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import chardet

from goapml.schemas import Action, ActionSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from goapml.models import PipelineConfig, WorldState


@dataclass(slots=True)
class DetectEncoding(Action):
    """Detect the CSV file's character encoding and record it on the state."""

    schema: ActionSchema = field(
        default_factory=lambda: ActionSchema(
            name="DetectEncoding",
            requires={"file_exists"},
            provides={"encoding_detected"},
            cost=1.0,
        ),
    )

    def run(self, state: WorldState, config: PipelineConfig) -> None:
        """Set ``state.encoding`` using explicit config, detection, or UTF-8 fallback.

        Always append a log line in the form ``encoding=...`` noting the source.
        """
        # Prefer explicit configuration when available
        explicit = config.file.encoding
        if explicit:
            state.encoding = explicit
            state.add("encoding_detected")
            state.logs.append(f"encoding={explicit} source=explicit")
            return

        # Read a reasonable chunk of the file to run detection on
        file_path = Path(config.file.path)
        # Safety: if precondition was bypassed, handle gracefully
        if not file_path.is_file():  # pragma: no cover - defensive guard
            state.encoding = "utf-8"
            state.add("encoding_detected")
            state.logs.append("encoding=utf-8 source=fallback_missing_file")
            return

        # Read up to 1MB for detection for a balance of speed/accuracy
        probe_bytes = file_path.read_bytes()[: 1024 * 1024]
        result = chardet.detect(probe_bytes)
        detected = result.get("encoding")

        if isinstance(detected, str) and detected:
            state.encoding = detected
            state.add("encoding_detected")
            conf = result.get("confidence")
            conf_info = f" confidence={float(conf):.3f}" if isinstance(conf, float) else ""
            state.logs.append(f"encoding={detected} source=detected{conf_info}")
            return

        # Fallback: default to UTF-8 when detection fails
        state.encoding = "utf-8"
        state.add("encoding_detected")
        state.logs.append("encoding=utf-8 source=fallback")

