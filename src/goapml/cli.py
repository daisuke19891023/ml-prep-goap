"""Typer-based command line interface for running the GOAP ML pipeline."""

from __future__ import annotations

import json
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError

from goapml.engine import ExecutionError, execute_with_replanning
from goapml.logging import configure_logging as configure_structured_logging
from goapml.models import (
    ArtifactSpec,
    EvalPolicy,
    FileSpec,
    ModelKind,
    ModelPolicy,
    PipelineConfig,
    SplitPolicy,
    TargetSpec,
    WorldState,
)
import goapml.paths as path_utils
from goapml.schemas import Goal

__all__ = ["app"]


MetricLiteral = Literal["r2", "rmse", "mae"]

DEFAULT_METRICS: tuple[MetricLiteral, ...] = ("r2", "rmse", "mae")


LOGGER = logging.getLogger(__name__)
class MetricName(StrEnum):
    """Evaluation metrics accepted by the CLI."""

    R2 = "r2"
    RMSE = "rmse"
    MAE = "mae"

    @property
    def literal(self) -> MetricLiteral:
        """Return the literal representation required by configuration models."""
        return self.value


class LogLevel(StrEnum):
    """Logging levels offered as CLI options."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Run the GOAP regression pipeline end-to-end from the command line.",
)


def _configure_logging(level: LogLevel) -> None:
    """Initialise basic logging according to the requested level."""
    mapping = logging.getLevelNamesMapping()
    numeric_level = mapping.get(level.value.upper(), logging.INFO)
    configure_structured_logging(numeric_level)


def _normalise_encoding(value: str | None) -> str | None:
    """Treat empty strings as missing encoding declarations."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _metrics_or_default(metrics: tuple[MetricName, ...] | None) -> list[MetricLiteral]:
    """Return the requested metrics or the default list when omitted."""
    if metrics is None or not metrics:
        return list(DEFAULT_METRICS)
    return [metric.literal for metric in metrics]


def _coerce_extra_metrics(values: tuple[str, ...]) -> tuple[MetricName, ...]:
    """Convert additional CLI args into ``MetricName`` values."""
    extras: list[MetricName] = []
    for arg in values:
        if arg == "run":
            continue
        if arg.startswith("--"):
            message = f"Unexpected option passed after metrics: {arg}"
            raise typer.BadParameter(message, param_hint="metrics")
        if arg.startswith("-"):
            message = f"Unexpected option passed after metrics: {arg}"
            raise typer.BadParameter(message, param_hint="metrics")
        try:
            extras.append(MetricName(arg))
        except ValueError as exc:
            message = f"Unsupported metric requested: {arg}"
            raise typer.BadParameter(message, param_hint="metrics") from exc
    return tuple(extras)


def _validate_json_out_path(
    json_out: Path | None, *, root: Path | None = None,
) -> Path | None:
    """Validate and normalise the requested JSON output path."""
    if json_out is None:
        return None

    param_hint = "--json-out"
    base = (root or Path.cwd()).resolve()
    candidate = Path(json_out)

    if ".." in candidate.parts:
        message = "parent directory references are not allowed in --json-out"
        raise typer.BadParameter(message, param_hint=param_hint)

    try:
        resolved = candidate.expanduser().resolve(strict=False)
    except OSError as exc:
        message = f"unable to resolve output path: {json_out}"
        raise typer.BadParameter(message, param_hint=param_hint) from exc

    try:
        resolved.relative_to(base)
    except ValueError as exc:
        message = "output path must reside within the current working directory"
        raise typer.BadParameter(message, param_hint=param_hint) from exc

    anchored = candidate if candidate.is_absolute() else base / candidate

    try:
        safe_path = path_utils.ensure_safe_path(base, anchored)
    except path_utils.UnsafePathError as exc:
        message = "symbolic links or reparse points are not permitted for --json-out"
        raise typer.BadParameter(message, param_hint=param_hint) from exc

    return safe_path


def _emit_json(payload: dict[str, object], json_out: Path | None) -> None:
    """Write ``payload`` to stdout and, optionally, a JSON file."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    typer.echo(text)
    if json_out is not None:
        if not path_utils.secure_open_supported():
            message = (
                "Secure JSON output (--json-out) is unsupported on this platform; "
                "os.O_NOFOLLOW and dir_fd support are required."
            )
            typer.echo(message, err=True)
            raise typer.Exit(code=1)

        target_path = Path(json_out)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = (
                "Unable to create directory for JSON output "
                f"({target_path.parent}): {exc}"
            )
            typer.echo(message, err=True)
            raise typer.Exit(code=1) from exc

        absolute_target = target_path.absolute()
        directory_parts = absolute_target.parts[:-1]
        file_name = absolute_target.name

        directory_flags = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        path_flag = getattr(os, "O_PATH", 0)
        if path_flag:
            directory_flags |= path_flag
        else:
            directory_flags |= os.O_RDONLY

        dir_fd: int | None = None
        try:
            for index, component in enumerate(directory_parts):
                if index == 0 and absolute_target.is_absolute():
                    path_component = component or os.sep
                    next_fd = os.open(path_component, directory_flags)
                else:
                    next_fd = os.open(component, directory_flags, dir_fd=dir_fd)
                    if dir_fd is not None:
                        os.close(dir_fd)
                dir_fd = next_fd

            file_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            file_flags |= getattr(os, "O_CLOEXEC", 0)
            file_descriptor = os.open(file_name, file_flags, 0o600, dir_fd=dir_fd)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as writer:
                writer.write(text + "\n")
        except OSError as exc:
            message = (
                "Refusing to write JSON output because the destination directory "
                "changed during execution (possible symlink)."
            )
            typer.echo(message, err=True)
            raise typer.Exit(code=1) from exc
        finally:
            if dir_fd is not None:
                os.close(dir_fd)


def _build_success_payload(
    *,
    state: WorldState,
    executed: list[str],
) -> dict[str, object]:
    """Assemble the JSON payload returned after successful execution."""
    metrics = state.metrics or {}
    return {
        "status": "ok",
        "actions": executed,
        "metrics": metrics,
        "logs": list(state.logs),
    }


def _build_failure_payload(
    *,
    message: str,
    state: WorldState,
) -> dict[str, object]:
    """Assemble the JSON payload returned when execution fails."""
    return {
        "status": "error",
        "error": message,
        "logs": list(state.logs),
    }


@app.command(context_settings={"allow_extra_args": True})
def run(
    ctx: typer.Context,
    *,
    csv: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            writable=False,
            readable=True,
            help="Path to the source CSV file.",
        ),
    ],
    target: Annotated[str, typer.Option(help="Name of the regression target column.")],
    encoding: Annotated[
        str,
        typer.Option(
            help="Explicit CSV encoding. Leave empty to auto-detect.",
        ),
    ] = "",
    delimiter: Annotated[
        str,
        typer.Option(help="Field delimiter used in the CSV file."),
    ] = ",",
    decimal: Annotated[
        str,
        typer.Option(help="Decimal separator used in the CSV file."),
    ] = ".",
    header: Annotated[
        bool,
        typer.Option(
            "--header/--no-header",
            help="Indicate whether the CSV contains a header row.",
        ),
    ] = True,
    model: Annotated[
        ModelKind,
        typer.Option(help="Regression model to train.", case_sensitive=False),
    ] = ModelKind.LINEAR_REGRESSION,
    test_size: Annotated[
        float,
        typer.Option(help="Validation split ratio between 0.05 and 0.5."),
    ] = 0.2,
    metrics: Annotated[
        MetricName | None,
        typer.Option(
            help="Evaluation metrics to compute.",
            metavar="METRIC",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(help="Optional path for saving the JSON output."),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option(help="Logging verbosity for the run.", case_sensitive=False),
    ] = LogLevel.INFO,
) -> None:
    """Run the GOAP regression pipeline and report evaluation metrics."""
    _configure_logging(log_level)
    safe_json_out = _validate_json_out_path(json_out)

    state = WorldState(facts={"file_exists"})
    goal = Goal(required={"evaluated"})

    option_metrics = (metrics,) if metrics is not None else ()
    extra_metrics = _coerce_extra_metrics(tuple(ctx.args))
    all_metrics = option_metrics + extra_metrics
    metrics_to_use = _metrics_or_default(all_metrics if all_metrics else None)

    try:
        csv_path = Path(csv)
        config = PipelineConfig(
            file=FileSpec(
                path=str(csv_path),
                encoding=_normalise_encoding(encoding),
                delimiter=delimiter,
                decimal=decimal,
                has_header=header,
            ),
            target=TargetSpec(strategy="explicit", name=target),
            split=SplitPolicy(test_size=test_size),
            model=ModelPolicy(kind=model, params={}),
            eval=EvalPolicy(metrics=metrics_to_use),
            artifacts_root=csv_path.parent,
            artifacts=ArtifactSpec(directory="artifacts"),
        )
    except ValidationError as exc:
        typer.echo("Configuration validation failed:", err=True)
        typer.echo(str(exc), err=True)
        payload = _build_failure_payload(message="configuration_error", state=state)
        if safe_json_out is not None:
            _emit_json(payload, safe_json_out)
        raise typer.Exit(code=1) from exc

    try:
        executed = execute_with_replanning(state=state, config=config, goal=goal)
    except ExecutionError as exc:
        typer.echo(f"Execution failed: {exc}", err=True)
        payload = _build_failure_payload(message=str(exc), state=state)
        _emit_json(payload, safe_json_out)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        LOGGER.exception("Unhandled exception during pipeline execution")
        payload = _build_failure_payload(message=str(exc), state=state)
        _emit_json(payload, safe_json_out)
        raise typer.Exit(code=1) from exc

    payload = _build_success_payload(state=state, executed=executed)
    _emit_json(payload, safe_json_out)


if __name__ == "__main__":  # pragma: no cover - script entry point
    app()
