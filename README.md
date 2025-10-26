# GOAP-Driven Regression Pipeline

This repository hosts an incremental build of a Goal-Oriented Action Planning (GOAP) system that automates a regression
workflow from CSV ingestion through evaluation. The implementation follows the plan captured in
[`docs/reference/plan-and-tasks.md`](docs/reference/plan-and-tasks.md).

## Current Status

- ✅ Project scaffold, documentation skeleton, and dependency pins (Task T1).
- ✅ Core GOAP components (configuration, world state, planner, actions, engine).
- ✅ Typer-powered CLI for executing the end-to-end pipeline (Task T20).

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

## CLI Usage

The `goapml` command executes the full CSV ➜ preprocessing ➜ training ➜ evaluation flow.

```bash
goapml run \
  --csv ./data.csv \
  --target SalePrice \
  --model random_forest \
  --test-size 0.2 \
  --metrics r2 rmse mae \
  --json-out result.json \
  --log-level INFO
```

The command prints metrics and action logs to stdout and writes the same payload to the optional JSON file.
If the destination directory for ``--json-out`` changes into a symbolic link while the command runs,
the CLI aborts instead of writing to an unexpected location.

## Documentation

- [GOAP Regression Pipeline Plan](docs/reference/plan-and-tasks.md)
- Additional architecture decision records will be added under `docs/adr/` as new components land.

## License

Distributed under the terms of the [MIT License](LICENSE).

