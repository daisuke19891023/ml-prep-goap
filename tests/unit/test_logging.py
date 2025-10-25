"""Tests for structured logging configuration utilities."""

from __future__ import annotations

import json
import logging

from goapml.logging import StructuredFormatter, configure_logging


def test_structured_formatter_renders_json_payload() -> None:
    """StructuredFormatter should include extras and basic fields."""
    record = logging.LogRecord(
        name="goapml.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Processing %s",
        args=("sample",),
        exc_info=None,
    )
    record.event = "unit-test"
    formatter = StructuredFormatter()

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "Processing sample"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "goapml.test"
    assert payload["event"] == "unit-test"


def test_configure_logging_installs_structured_handler() -> None:
    """configure_logging should replace root handlers with structured formatter."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    for handler in list(root.handlers):
        root.removeHandler(handler)

    try:
        configure_logging(logging.INFO)
        configured_root = logging.getLogger()
        assert configured_root.level == logging.INFO
        assert any(
            isinstance(handler.formatter, StructuredFormatter)
            for handler in configured_root.handlers
            if handler.formatter is not None
        )
    finally:
        for handler in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
