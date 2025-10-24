# GOAP-Driven Regression Pipeline

This repository hosts an incremental build of a Goal-Oriented Action Planning (GOAP) system that automates a regression
workflow from CSV ingestion through evaluation. The implementation follows the plan captured in
[`docs/reference/plan-and-tasks.md`](docs/reference/plan-and-tasks.md).

## Current Status

- ✅ Project scaffold, documentation skeleton, and dependency pins (Task T1).
- ✅ Pydantic-based `PipelineConfig` definitions with validation (Task T2).
- ⬜ Subsequent tasks (world state, planner, actions, engine, CLI) pending.

## Development Setup

```bash
uv sync --extra dev
```

Run quality checks on each edit:

```bash
uv run nox -s lint
uv run nox -s typing
uv run nox -s test
```

## Documentation

- [GOAP Regression Pipeline Plan](docs/reference/plan-and-tasks.md)
- Additional architecture decision records will be added under `docs/adr/` as new components land.

## License

Distributed under the terms of the [MIT License](LICENSE).

