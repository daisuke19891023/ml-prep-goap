# GOAP Regression Pipeline Plan

This document records the high-level architecture and milestone plan for building a Goal-Oriented Action Planning (GOAP)
workflow that automates a CSV ➜ preprocessing ➜ regression training ➜ prediction ➜ evaluation pipeline.

## Final Goal

*Goal fact:* `evaluated`

The system must load a CSV dataset, identify and validate the regression target column, run preprocessing, train a model, make
predictions for a validation split, and compute regression metrics such as R², RMSE, and MAE.

## Architecture Summary

- **Configuration**: `PipelineConfig` (Pydantic v2) describes all declarative inputs including file metadata, preprocessing
  policies, model selection, evaluation metrics, and planner behaviour.
- **World State**: runtime artefacts (DataFrame, encoders, model) plus factual flags that the GOAP planner consumes.
- **Actions**: each processing step exposes a schema (`requires` / `provides` facts and a `cost`) and an implementation that
  mutates the world state.
- **Planner**: A* search over fact sets builds a minimal-cost plan leading to `evaluated`.
- **Engine**: executes the plan, replanning on recoverable failures.

## Task Breakdown

| Task | Title | Summary | Status |
| --- | --- | --- | --- |
| T1 | Scaffold & Dependencies | Create the project layout, docs skeleton, and dependency pins. | ✅ Completed |
| T2 | PipelineConfig (Pydantic) | Implement configuration models and validators. | ✅ Completed |
| T3 | WorldState | Model runtime artefacts and convenience helpers. | ⬜ Pending |
| T4 | ActionSchema / Action / Goal | Describe action contracts. | ⬜ Pending |
| T5 | A* Planner | Implement fact-based planner with heuristic search. | ⬜ Pending |
| T6 | Action Registry | Registry for schema/implementation binding. | ⬜ Pending |
| T7–T17 | Action Implementations | Concrete steps from I/O through evaluation. | ⬜ Pending |
| T18 | Execution Engine | Replanning executor orchestration. | ⬜ Pending |
| T19 | Action Declarations | Register standard action schemas. | ⬜ Pending |
| T20 | CLI | Typer-based interface for running the pipeline. | ⬜ Pending |
| T21 | End-to-End Test | Synthetic dataset regression validation. | ⬜ Pending |
| T22 | Logging | Structured logging integration. | ✅ Completed |
| T23 | Type & Lint | Static analysis and formatting enforcement. | ⬜ Pending |
| T24 | Persist Artifacts (optional) | Persist model and preprocessor with joblib. | ⬜ Pending |
| T25 | Model Selection (optional) | Search over hyper-parameters. | ⬜ Pending |

## Milestones

- **M0 – Scaffold**: Directory layout, docs, dependency management. *(✅ completed)*
- **M1 – Models & Schemas**: Config and world state Pydantic models.
- **M2 – Planner**: A* search across factual states.
- **M3 – Actions**: Implement action modules (I/O, preprocessing, training, evaluation).
- **M4 – Engine & Registry**: Execution flow with replanning.
- **M5 – CLI & E2E**: User-facing entry point and integration test.
- **M6 – Quality**: Logging, linting, typing, formatting automation.
- **M7 – Extensions**: Optional persistence or model selection features.

## Acceptance Criteria Snapshot

- Dependencies are managed via `uv` and pinned in `pyproject.toml`.
- `PipelineConfig` validates CSV availability, target strategy, preprocessing policies, and planner options.
- Subsequent tasks will introduce world state tracking, GOAP schema definitions, the planner, and action implementations.

