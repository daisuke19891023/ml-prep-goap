"""Typer-based command line interface for running the GOAP ML pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from enum import StrEnum
from typing import Annotated, Literal
import typer
from pydantic import ValidationError

from goapml.engine import ExecutionError, execute_with_replanning
from goapml.logging import configure_logging as configure_structured_logging
from goapml.models import (
    EvalPolicy,
    FileSpec,
    ModelPolicy,
    PipelineConfig,
    SplitPolicy,
    TargetSpec,
    WorldState,
)
from goapml.schemas import Goal

__all__ = ["app"]


MetricLiteral = Literal["r2", "rmse", "mae"]

DEFAULT_METRICS: tuple[MetricLiteral, ...] = ("r2", "rmse", "mae")


LOGGER = logging.getLogger(__name__)


class ModelKind(StrEnum):
    """Supported regression model identifiers exposed via the CLI."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"


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


def _emit_json(payload: dict[str, object], json_out: Path | None) -> None:
    """Write ``payload`` to stdout and, optionally, a JSON file."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    typer.echo(text)
    if json_out is not None:
        target_path = Path(json_out)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(text + "\n", encoding="utf-8")


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
            model=ModelPolicy(kind=model.value, params={}),
            eval=EvalPolicy(metrics=metrics_to_use),
        )
    except ValidationError as exc:
        typer.echo("Configuration validation failed:", err=True)
        typer.echo(str(exc), err=True)
        payload = _build_failure_payload(message="configuration_error", state=state)
        if json_out is not None:
            _emit_json(payload, json_out)
        raise typer.Exit(code=1) from exc

    try:
        executed = execute_with_replanning(state=state, config=config, goal=goal)
    except ExecutionError as exc:
        typer.echo(f"Execution failed: {exc}", err=True)
        payload = _build_failure_payload(message=str(exc), state=state)
        _emit_json(payload, json_out)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - unexpected failures
        LOGGER.exception("Unhandled exception during pipeline execution")
        payload = _build_failure_payload(message=str(exc), state=state)
        _emit_json(payload, json_out)
        raise typer.Exit(code=1) from exc

    payload = _build_success_payload(state=state, executed=executed)
    _emit_json(payload, json_out)


if __name__ == "__main__":  # pragma: no cover - script entry point
    app()
