"""Tests for training and prediction actions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from goapml.actions.train import PersistArtifacts, Predict, TrainModel
from goapml.models import (
    ArtifactSpec,
    FileSpec,
    ModelKind,
    ModelPolicy,
    PipelineConfig,
    WorldState,
)
from pydantic import ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    def _joblib_load(
        filename: str,
        mmap_mode: str | None = None,
        ensure_native_byte_order: str = "auto",
    ) -> Any: ...
else:  # pragma: no cover - runtime import
    from joblib import load as _joblib_load


@pytest.fixture
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Return a minimal pipeline configuration for training actions."""
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("feature,target\n1,1\n", encoding="utf-8")

    file_spec = FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )

    return PipelineConfig(
        file=file_spec,
        artifacts_root=tmp_path,
        artifacts=ArtifactSpec(directory="artifacts"),
    )


def _build_file_spec(tmp_path: Path) -> FileSpec:
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("feature,target\n1,1\n", encoding="utf-8")
    return FileSpec(
        path=str(csv_path),
        encoding=None,
        delimiter=",",
        decimal=".",
        has_header=True,
    )


def build_state() -> WorldState:
    """Construct a world state with preprocessed train/test arrays."""
    x_train = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)
    return WorldState(split=(x_train, x_test, y_train, y_test), facts={"features_ready"})


def test_artifact_directory_rejects_absolute_path(tmp_path: Path) -> None:
    """Absolute artefact directories are rejected during validation."""
    file_spec = _build_file_spec(tmp_path)

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=file_spec,
            artifacts_root=tmp_path,
            artifacts=ArtifactSpec(directory=str(tmp_path / "absolute")),
        )

    assert "relative path" in str(exc.value)


def test_artifact_directory_rejects_parent_escapes(tmp_path: Path) -> None:
    """Directories using ``..`` are refused."""
    file_spec = _build_file_spec(tmp_path)

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=file_spec,
            artifacts_root=tmp_path,
            artifacts=ArtifactSpec(directory="../escape"),
        )

    assert "must not contain '..'" in str(exc.value)


def test_artifact_directory_rejects_symlinks(tmp_path: Path) -> None:
    """Symlinked artefact directories escaping the root are rejected."""
    file_spec = _build_file_spec(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    link = root / "link"
    target = tmp_path.parent
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform without symlink support
        pytest.skip(f"symlink not supported on this platform: {exc}")

    with pytest.raises(ValidationError) as exc:
        PipelineConfig(
            file=file_spec,
            artifacts_root=root,
            artifacts=ArtifactSpec(directory="link"),
        )

    assert "escapes the configured output root" in str(exc.value)


def test_resolve_artifact_paths_returns_safe_locations(
    pipeline_config: PipelineConfig,
) -> None:
    """Resolved artefact paths remain under the configured root."""
    directory, model_path, preprocessor_path = pipeline_config.resolve_artifact_paths()
    root = Path(pipeline_config.artifacts_root).resolve()

    assert directory.is_absolute()
    assert directory.parent == root
    assert model_path.parent == directory
    assert model_path.name == pipeline_config.artifacts.model_filename
    assert preprocessor_path.parent == directory
    assert (
        preprocessor_path.name == pipeline_config.artifacts.preprocessor_filename
    )


def test_persist_artifacts_rejects_symlink_targets(
    pipeline_config: PipelineConfig,
) -> None:
    """Symlinked artefact files are not overwritten."""
    x_train = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)

    scaler: Any = StandardScaler()
    scaler.fit(x_train)

    state = WorldState(
        split=(x_train, x_test, y_train, y_test),
        preprocessor=scaler,
        facts={"features_ready"},
    )

    TrainModel().run(state, pipeline_config)

    directory, model_path, _ = pipeline_config.resolve_artifact_paths()
    directory.mkdir(parents=True, exist_ok=True)
    outside = Path(pipeline_config.artifacts_root).resolve().parent / "outside.joblib"
    outside.write_bytes(b"outside")
    try:
        model_path.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform without symlink support
        pytest.skip(f"symlink not supported on this platform: {exc}")

    with pytest.raises(RuntimeError) as exc:
        PersistArtifacts().run(state, pipeline_config)

    assert "Failed to persist artefact" in str(exc.value)


def test_persist_artifacts_rejects_platform_without_secure_open(
    pipeline_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Platforms without secure os.open support should be rejected."""
    x_train = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)

    scaler: Any = StandardScaler()
    scaler.fit(x_train)

    state = WorldState(
        split=(x_train, x_test, y_train, y_test),
        preprocessor=scaler,
        facts={"features_ready"},
    )

    TrainModel().run(state, pipeline_config)

    monkeypatch.setattr("goapml.paths.secure_open_supported", lambda: False)

    with pytest.raises(RuntimeError) as exc:
        PersistArtifacts().run(state, pipeline_config)

    assert "Artifact persistence is unsupported" in str(exc.value)


def test_persist_artifacts_rejects_directory_symlink_swap(
    pipeline_config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symlink swaps of the artefact directory are rejected."""
    x_train = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)

    scaler: Any = StandardScaler()
    scaler.fit(x_train)

    state = WorldState(
        split=(x_train, x_test, y_train, y_test),
        preprocessor=scaler,
        facts={"features_ready"},
    )

    TrainModel().run(state, pipeline_config)

    directory, model_path, preprocessor_path = pipeline_config.resolve_artifact_paths()
    original_resolve = PipelineConfig.resolve_artifact_paths

    def _patched_resolve(self: PipelineConfig) -> tuple[Path, Path, Path]:
        if self is pipeline_config:
            return directory, model_path, preprocessor_path
        return original_resolve(self)

    monkeypatch.setattr(PipelineConfig, "resolve_artifact_paths", _patched_resolve)
    directory.mkdir(parents=True, exist_ok=True)
    outside_root = Path(pipeline_config.artifacts_root).resolve().parent / "outside"
    outside_root.mkdir(parents=True, exist_ok=True)

    try:
        directory.rmdir()
        directory.symlink_to(outside_root, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform without symlink support
        pytest.skip(f"symlink not supported on this platform: {exc}")

    with pytest.raises(RuntimeError) as exc:
        PersistArtifacts().run(state, pipeline_config)

    assert "Failed to persist artefact directory" in str(exc.value)
    assert not (outside_root / model_path.name).exists()
    assert not (outside_root / preprocessor_path.name).exists()


def test_train_model_linear_regression_fits_and_logs(
    pipeline_config: PipelineConfig,
) -> None:
    """Linear regression is fitted and recorded on the state."""
    state = build_state()

    TrainModel().run(state, pipeline_config)

    assert state.model is not None
    from sklearn.linear_model import LinearRegression

    assert isinstance(state.model, LinearRegression)
    assert state.has("trained")
    assert state.logs[-1] == "train_model:linear_regression"


def test_train_model_supports_random_forest(pipeline_config: PipelineConfig) -> None:
    """Random forest models are instantiated using provided parameters."""
    state = build_state()
    model_policy = ModelPolicy(
        kind=ModelKind.RANDOM_FOREST,
        params={"n_estimators": 5, "random_state": 0},
    )
    config = pipeline_config.model_copy(update={"model": model_policy})

    TrainModel().run(state, config)

    from sklearn.ensemble import RandomForestRegressor

    assert isinstance(state.model, RandomForestRegressor)
    assert state.has("trained")


def test_predict_generates_predictions_matching_target(
    pipeline_config: PipelineConfig,
) -> None:
    """Predictions align with the validation target size."""
    state = build_state()
    TrainModel().run(state, pipeline_config)

    Predict().run(state, pipeline_config)

    assert state.pred is not None
    assert state.split is not None
    _, _, _, y_test = state.split
    assert len(state.pred) == len(y_test)
    assert state.has("predicted")
    assert state.logs[-1] == "predict"


def test_persist_artifacts_round_trip_prediction(
    pipeline_config: PipelineConfig,
) -> None:
    """Persisted artefacts can be reloaded to reproduce predictions."""
    x_train_raw = np.array([[1.0], [2.0], [3.0]], dtype=float)
    x_test_raw = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)

    scaler: Any = StandardScaler()
    x_train_processed = cast(
        "NDArray[np.float64]",
        np.asarray(scaler.fit_transform(x_train_raw), dtype=float),
    )
    x_test_processed = cast(
        "NDArray[np.float64]",
        np.asarray(scaler.transform(x_test_raw), dtype=float),
    )

    state = WorldState(
        split=(x_train_processed, x_test_processed, y_train, y_test),
        preprocessor=scaler,
        facts={"features_ready"},
    )

    TrainModel().run(state, pipeline_config)
    Predict().run(state, pipeline_config)

    PersistArtifacts().run(state, pipeline_config)

    assert state.has("persisted")
    assert state.model_path is not None
    assert state.preprocessor_path is not None
    assert state.logs[-1] == "persist_artifacts:model.joblib,preprocessor.joblib"

    loaded_model = _joblib_load(str(state.model_path))
    loaded_preprocessor = _joblib_load(str(state.preprocessor_path))

    transformed_test = cast(
        "NDArray[np.float64]",
        loaded_preprocessor.transform(x_test_raw),
    )
    reloaded_predictions = cast(
        "NDArray[np.float64]",
        loaded_model.predict(transformed_test),
    )

    assert state.pred is not None
    baseline_pred = np.asarray(state.pred, dtype=float)
    np.testing.assert_allclose(reloaded_predictions, baseline_pred)


def test_persist_artifacts_requires_preprocessor(
    pipeline_config: PipelineConfig,
) -> None:
    """Persisting artefacts without a preprocessor raises an error."""
    state = build_state()
    TrainModel().run(state, pipeline_config)

    with pytest.raises(RuntimeError):
        PersistArtifacts().run(state, pipeline_config)
