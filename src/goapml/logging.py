"""Utilities for configuring structured logging across the project."""

from __future__ import annotations

import json
import logging
from logging.config import dictConfig
from typing import Any

__all__ = ["StructuredFormatter", "configure_logging"]


_STANDARD_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class StructuredFormatter(logging.Formatter):
    """Render log records as single-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert a record to JSON, preserving extra fields."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_")
        }
        payload.update(extras)

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(level: int | str) -> None:
    """Initialise the root logger with structured console output."""
    if isinstance(level, str):
        mapping = logging.getLevelNamesMapping()
        numeric_level = mapping.get(level.upper(), logging.INFO)
    else:
        numeric_level = int(level)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "()": StructuredFormatter,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": numeric_level,
                    "formatter": "structured",
                    "stream": "ext://sys.stderr",
                },
            },
            "root": {
                "level": numeric_level,
                "handlers": ["console"],
            },
        },
    )
