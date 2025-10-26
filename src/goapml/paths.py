"""Utilities for validating secure filesystem paths."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import cast

__all__ = [
    "UnsafePathError",
    "ensure_safe_path",
    "secure_open_supported",
]


class UnsafePathError(RuntimeError):
    """Raised when a filesystem path traverses unsafe components."""


def _stat_path(path: Path) -> None:
    """Validate that ``path`` is not a symbolic link or reparse point."""
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        # Non-existent intermediate paths will be created later; they cannot be
        # validated yet but also cannot be malicious symlinks at this stage.
        return
    except OSError as exc:  # pragma: no cover - propagated for clarity
        message = f"unable to stat path component: {path}"
        raise UnsafePathError(message) from exc

    if stat.S_ISLNK(info.st_mode):
        message = f"encountered symbolic link: {path}"
        raise UnsafePathError(message)

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(info, "st_file_attributes", 0)
    if reparse_flag and file_attributes & reparse_flag:
        message = f"encountered reparse point: {path}"
        raise UnsafePathError(message)


def ensure_safe_path(base: Path, target: Path) -> Path:
    """Ensure ``target`` resides under ``base`` without unsafe components."""
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = (Path.cwd() / base_path).resolve()

    try:
        base_stat = base_path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        message = f"base directory does not exist: {base_path}"
        raise UnsafePathError(message) from exc
    except OSError as exc:  # pragma: no cover - escalated for observability
        message = f"unable to stat base directory: {base_path}"
        raise UnsafePathError(message) from exc

    if stat.S_ISLNK(base_stat.st_mode):
        message = f"base directory is a symbolic link: {base_path}"
        raise UnsafePathError(message)

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(base_stat, "st_file_attributes", 0)
    if reparse_flag and file_attributes & reparse_flag:
        message = f"base directory is a reparse point: {base_path}"
        raise UnsafePathError(message)

    candidate = target if target.is_absolute() else base_path / target

    try:
        relative = candidate.relative_to(base_path)
    except ValueError as exc:
        message = f"target escapes the base directory: {candidate}"
        raise UnsafePathError(message) from exc

    current = base_path
    for part in relative.parts:
        current = current / part
        _stat_path(current)

    return candidate


def secure_open_supported() -> bool:
    """Return whether secure ``os.open`` flags are available."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if hasattr(os, "supports_dir_fd"):
        support_container = cast("set[object]", os.supports_dir_fd)
        try:
            os_open = os.open
        except AttributeError:  # pragma: no cover - defensive guard
            os_open = None
        dir_fd_support = "open" in support_container
        if os_open is not None:
            dir_fd_support = dir_fd_support or os_open in support_container
    else:
        dir_fd_support = False
    return bool(nofollow) and dir_fd_support

