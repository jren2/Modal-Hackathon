#!/usr/bin/env python3
"""Isolated Modal runner for the EgoTrim diversity V2 experiment.

Example:

    python -m modal run experiments/egotrim_diversity_v2/modal_app.py \
        --segments /egotrim-data/path/to/segments.csv \
        --pose-root /egotrim-data/path/to/poses \
        --smoke

Or adapt fixed-window model clips already stored beside EgoVerse episodes:

    python -m modal run experiments/egotrim_diversity_v2/modal_app.py \
        --egoverse-clips-root /egoverse/segments \
        --egoverse-episodes-root /egoverse/episodes \
        --smoke

The source volume is mounted read-only. Every persisted artifact is written beneath
``/egotrim-models/egotrim-diversity-v2/<run_id>/``; an existing run directory is
never reused.
"""

from __future__ import annotations

import datetime as dt
import csv
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

import modal


APP_NAME = "egotrim-diversity-v2"
DATA_MOUNT = "/egotrim-data"
EGOVERSE_MOUNT = "/egoverse"
MODELS_MOUNT = "/egotrim-models"
RUNS_ROOT = "/egotrim-models/egotrim-diversity-v2"
CORE_REMOTE_DIR = "/opt/egotrim-diversity-v2"
MAX_SMOKE_EPISODES = 3

LOCAL_ROOT = Path(__file__).resolve().parent
CORE_LOCAL_PATH = LOCAL_ROOT / "run_egotrim_diversity_v2.py"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(
    "egotrim-data",
    create_if_missing=False,
)

models_volume = modal.Volume.from_name(
    "egotrim-models",
    create_if_missing=False,
)

egoverse_volume = modal.Volume.from_name(
    "egoverse-zarrs-v2",
    create_if_missing=False,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2.0,<3",
        "pandas>=2.2,<4",
        "pyarrow>=16,<24",
        "scipy>=1.13,<2",
        "scikit-learn>=1.5,<2",
        "joblib>=1.4,<2",
        "matplotlib>=3.9,<4",
        "zarr>=3.0,<4",
    )
    .add_local_file(
        CORE_LOCAL_PATH,
        f"{CORE_REMOTE_DIR}/run_egotrim_diversity_v2.py",
        copy=True,
    )
)

FUNCTION_OPTIONS = {
    "image": image,
    "volumes": {
        DATA_MOUNT: data_volume.read_only(),
        EGOVERSE_MOUNT: egoverse_volume.read_only(),
        MODELS_MOUNT: models_volume,
    },
    "timeout": 60 * 60,
}


def _core() -> Any:
    """Import the experiment core bundled into the Modal image."""

    import importlib.util
    import sys

    path = Path(CORE_REMOTE_DIR) / "run_egotrim_diversity_v2.py"
    module_name = "egotrim_diversity_v2_core"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import bundled EgoTrim core from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/Pandas values before writing manifests."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen (maximum 128 characters)"
        )
    if run_id in {".", ".."}:
        raise ValueError("run_id may not be '.' or '..'")
    return Path(RUNS_ROOT) / run_id


def _mounted_path(
    raw_path: str,
    *,
    expect_directory: bool,
    allowed_roots: Iterable[str],
) -> Path:
    """Resolve an input while proving it remains inside an allowed mount."""

    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(
            f"Modal paths must be absolute; received {raw_path!r}"
        )
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        kind = "directory" if expect_directory else "file"
        raise FileNotFoundError(
            f"Expected mounted {kind}, but could not resolve {raw_path!r}"
        ) from exc
    resolved_roots: list[Path] = []
    for root in allowed_roots:
        try:
            resolved_roots.append(Path(root).resolve(strict=True))
        except FileNotFoundError:
            continue
    if not any(
        resolved == root or root in resolved.parents
        for root in resolved_roots
    ):
        raise ValueError(
            f"Path {resolved} is outside allowed mounts: {list(allowed_roots)}"
        )
    if expect_directory and not resolved.is_dir():
        raise NotADirectoryError(f"Pose root is not a directory: {resolved}")
    if not expect_directory and not resolved.is_file():
        raise FileNotFoundError(f"Segment manifest is not a file: {resolved}")
    return resolved


def _source_path(raw_path: str, *, expect_directory: bool) -> Path:
    """Resolve a read-only source from either supported data volume."""

    return _mounted_path(
        raw_path,
        expect_directory=expect_directory,
        allowed_roots=(DATA_MOUNT, EGOVERSE_MOUNT),
    )


def _input_manifest_path(raw_path: str) -> Path:
    """Resolve a source manifest or a generated manifest inside the run root."""

    return _mounted_path(
        raw_path,
        expect_directory=False,
        allowed_roots=(DATA_MOUNT, EGOVERSE_MOUNT, RUNS_ROOT),
    )


def _create_run_output(run_id: str) -> Path:
    output = _run_path(run_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / "features" / "episodes").mkdir(parents=True, exist_ok=False)
    (output / "baselines" / "jobs").mkdir(parents=True, exist_ok=False)
    return output


def _zarr_root_attributes(episode_path: Path) -> dict[str, Any]:
    metadata_path = episode_path / "zarr.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"EgoVerse episode is missing zarr.json: {episode_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    attributes = payload.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError(f"Invalid Zarr root attributes in {metadata_path}")
    return attributes


def _egoverse_clip_rows(
    clips_root: Path,
    episodes_root: Path,
    canonical_verb_override: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt EgoVerse fixed-window clip manifests to the core segment contract."""

    manifest_paths = sorted(clips_root.glob("*/*/manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(
            f"No clip manifests matching <episode>/<camera>/manifest.json under {clips_root}"
        )
    rows: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise ValueError(f"Invalid EgoVerse clip manifest: {manifest_path}")
        episode_id = str(payload.get("episode") or manifest_path.parent.parent.name).strip()
        if not episode_id:
            raise ValueError(f"Clip manifest does not identify an episode: {manifest_path}")
        episode_path = (episodes_root / episode_id).resolve(strict=True)
        if not episode_path.is_dir():
            raise NotADirectoryError(f"EgoVerse pose store is not a directory: {episode_path}")
        attributes = _zarr_root_attributes(episode_path)
        task_name = str(attributes.get("task_name", "")).strip().lower()
        canonical_verb = canonical_verb_override.strip().lower() or task_name
        if not canonical_verb:
            raise ValueError(
                f"Cannot derive a canonical task label for episode {episode_id}; "
                "pass --canonical-verb explicitly"
            )
        camera = manifest_path.parent.name
        fps = float(payload.get("fps", attributes.get("fps", 0.0)))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS in {manifest_path}: {fps}")
        episode_row_count = 0
        for clip in payload["segments"]:
            if not isinstance(clip, dict):
                raise ValueError(f"Non-object segment record in {manifest_path}")
            filename = str(clip.get("file", "")).strip()
            if not filename:
                raise ValueError(f"Segment record without a file in {manifest_path}")
            try:
                clip_path = (manifest_path.parent / filename).resolve(strict=True)
                clip_path.relative_to(manifest_path.parent.resolve(strict=True))
            except (FileNotFoundError, ValueError) as exc:
                raise FileNotFoundError(
                    f"Clip path is missing or escapes its manifest directory: {filename!r}"
                ) from exc
            start_time = float(clip["start_seconds"])
            duration = float(clip["duration_seconds"])
            start_frame = int(clip["start_frame"])
            end_frame_exclusive = int(clip["end_frame_exclusive"])
            if not all(math.isfinite(value) for value in (start_time, duration)):
                raise ValueError(f"Non-finite clip timing in {manifest_path}: {filename}")
            if start_time < 0 or duration <= 0:
                raise ValueError(f"Invalid clip timing in {manifest_path}: {filename}")
            zarr_total_frames = int(attributes.get("total_frames", 0))
            if (
                start_frame < 0
                or end_frame_exclusive <= start_frame
                or end_frame_exclusive > zarr_total_frames
            ):
                raise ValueError(
                    f"Clip frame bounds [{start_frame}, {end_frame_exclusive}) are outside "
                    f"the {zarr_total_frames}-frame episode: {filename}"
                )
            segment_id = f"{episode_id}:{camera}:{Path(filename).stem}"
            if segment_id in observed_ids:
                raise ValueError(f"Duplicate generated segment ID: {segment_id}")
            observed_ids.add(segment_id)
            rows.append(
                {
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "canonical_verb": canonical_verb,
                    "start_time": start_time,
                    "end_time": start_time + duration,
                    "duration": duration,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame_exclusive,
                    # Clip manifests do not declare pose validity. A neutral preflight
                    # value keeps the segment eligible; extraction measures real validity.
                    "tracking_valid_frac": 1.0,
                    "pose_ref": "",
                    "video_path": str(clip_path),
                    "source_data_path": str(episode_path),
                }
            )
            episode_row_count += 1
        episode_reports.append(
            {
                "episode_id": episode_id,
                "camera": camera,
                "manifest_path": str(manifest_path),
                "pose_path": str(episode_path),
                "task_name": task_name,
                "canonical_verb": canonical_verb,
                "clip_count": episode_row_count,
                "fps": fps,
                "zarr_total_frames": int(attributes.get("total_frames", 0)),
            }
        )
    return rows, {
        "adapter": "egoverse_fixed_window_clip_manifests_v1",
        "input_semantics": (
            "Fixed-window model clips; these are not semantic action-cycle annotations."
        ),
        "tracking_valid_frac": (
            "Not present in clip manifests; initialized to 1.0 only for preflight and "
            "replaced by measured pose validity during feature extraction."
        ),
        "manifest_count": len(manifest_paths),
        "clip_count": len(rows),
        "episodes": episode_reports,
    }


def _artifact_stem(episode_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("._-") or "episode"
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{safe[:80]}"


def _episode_rows(segments_path: Path, episode_id: str) -> tuple[Any, Any, Any]:
    core = _core()
    original = core.load_segments(segments_path)
    schema = core.discover_segment_schema(original)
    canonical = core.canonicalize_segments(original, schema)
    rows = canonical.loc[canonical["episode_id"].astype(str) == str(episode_id)].copy()
    if rows.empty:
        raise RuntimeError(f"Episode {episode_id!r} is absent from {segments_path}")
    return core, schema, rows


@app.function(**FUNCTION_OPTIONS)
def initialize_run(
    run_id: str,
    segments: str,
    pose_root: str,
    smoke: bool,
    budget_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Validate source data and create a never-before-used output directory."""

    if not 0.0 < budget_frac <= 1.0:
        raise ValueError("budget_frac must be in (0, 1]")
    segments_path = _source_path(segments, expect_directory=False)
    pose_path = _source_path(pose_root, expect_directory=True)
    core = _core()
    original = core.load_segments(segments_path)
    schema = core.discover_segment_schema(original)
    canonical = core.canonicalize_segments(original, schema)
    episode_ids = sorted(canonical["episode_id"].astype(str).unique().tolist())
    if smoke:
        episode_ids = episode_ids[:MAX_SMOKE_EPISODES]
    if not episode_ids:
        raise RuntimeError(f"No episodes were found in segment manifest {segments_path}")

    output = _create_run_output(run_id)
    manifest = {
        "status": "initialized",
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "segments": str(segments_path),
        "pose_root": str(pose_path),
        "smoke": bool(smoke),
        "smoke_episode_limit": MAX_SMOKE_EPISODES if smoke else None,
        "episode_ids": episode_ids,
        "episode_count": len(episode_ids),
        "budget_frac": float(budget_frac),
        "seed": int(seed),
        "segment_columns_observed": [str(column) for column in original.columns],
        "segment_schema": {
            key: value for key, value in vars(schema).items()
        },
        "source_volume": f"read-only source containing {segments_path}",
        "models_volume": "egotrim-models",
        "output_path": str(output),
    }
    _write_json_exclusive(output / "modal_run_manifest.json", manifest)
    models_volume.commit()
    return manifest


@app.function(**FUNCTION_OPTIONS)
def initialize_egoverse_run(
    run_id: str,
    clips_root: str,
    episodes_root: str,
    canonical_verb: str,
    smoke: bool,
    budget_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Adapt live EgoVerse clip windows and initialize a never-reused run."""

    if not 0.0 < budget_frac <= 1.0:
        raise ValueError("budget_frac must be in (0, 1]")
    clips_path = _source_path(clips_root, expect_directory=True)
    pose_path = _source_path(episodes_root, expect_directory=True)
    rows, adapter_report = _egoverse_clip_rows(
        clips_path,
        pose_path,
        canonical_verb,
    )
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    if smoke:
        episode_ids = episode_ids[:MAX_SMOKE_EPISODES]
    if not episode_ids:
        raise RuntimeError(f"No episodes were found beneath {clips_path}")

    output = _create_run_output(run_id)
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    segments_path = inputs / "egoverse_clip_segments.csv"
    fieldnames = list(rows[0])
    with segments_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    core = _core()
    original = core.load_segments(segments_path)
    schema = core.discover_segment_schema(original)
    core.canonicalize_segments(original, schema)
    manifest = {
        "status": "initialized",
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "segments": str(segments_path),
        "pose_root": str(pose_path),
        "clips_root": str(clips_path),
        "smoke": bool(smoke),
        "smoke_episode_limit": MAX_SMOKE_EPISODES if smoke else None,
        "episode_ids": episode_ids,
        "episode_count": len(episode_ids),
        "budget_frac": float(budget_frac),
        "seed": int(seed),
        "segment_columns_observed": fieldnames,
        "segment_schema": vars(schema),
        "input_adapter": adapter_report,
        "source_volume": "egoverse-zarrs-v2 (read-only mount)",
        "models_volume": "egotrim-models",
        "output_path": str(output),
    }
    _write_json_exclusive(output / "modal_run_manifest.json", manifest)
    models_volume.commit()
    return manifest


@app.function(**FUNCTION_OPTIONS, max_containers=64, memory=4096)
def extract_episode_features(
    run_id: str,
    episode_id: str,
    segments: str,
    pose_root: str,
    max_interp_gap: int = 2,
    coordinate_fallback: str = "error",
) -> dict[str, Any]:
    """Extract and persist raw balanced-feature blocks for one episode."""

    import numpy as np

    output = _run_path(run_id)
    if not output.is_dir():
        raise RuntimeError(f"Run was not initialized: {output}")
    segments_path = _input_manifest_path(segments)
    pose_path = _source_path(pose_root, expect_directory=True)
    core, schema, rows = _episode_rows(segments_path, episode_id)
    trajectories: list[Any] = []
    fingertips: list[Any] = []
    log_durations: list[float] = []
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    pose_examples: list[dict[str, Any]] = []

    for row in rows.to_dict(orient="records"):
        base = dict(row)
        base.update(
            {
                "is_valid": False,
                "rejection_reason": "",
                "tracking_quality": None,
                "measured_tracking_valid_frac": None,
                "missing_value_rate": None,
                "interpolated_value_count": None,
                "coordinate_mode": None,
                "pose_source_path": None,
            }
        )
        if float(row["duration"]) < core.MIN_DURATION_SECONDS:
            base["rejection_reason"] = (
                f"duration_below_{core.MIN_DURATION_SECONDS}_seconds"
            )
            rejected_rows.append(base)
            continue
        if float(row["tracking_valid_frac"]) < core.MIN_TRACKING_VALID_FRAC:
            base["rejection_reason"] = (
                f"declared_tracking_valid_frac_below_{core.MIN_TRACKING_VALID_FRAC}"
            )
            rejected_rows.append(base)
            continue
        try:
            pose = core.load_pose_segment(row, pose_path, coordinate_fallback)
            feature = core.extract_raw_feature(
                pose,
                float(row["duration"]),
                max_interp_gap,
                core.MIN_TRACKING_VALID_FRAC,
            )
        except core.TrackingError as exc:
            base["rejection_reason"] = f"tracking_rejected:{exc}"
            rejected_rows.append(base)
            continue

        valid = dict(row)
        valid.update(
            {
                "measured_tracking_valid_frac": feature.measured_tracking_valid_frac,
                "tracking_quality": min(
                    float(row["tracking_valid_frac"]),
                    feature.measured_tracking_valid_frac,
                ),
                "missing_value_rate": feature.missing_value_rate,
                "interpolated_value_count": feature.interpolated_value_count,
                "coordinate_mode": feature.coordinate_mode,
                "pose_source_path": feature.pose_source_path,
            }
        )
        if not str(valid.get("source_data_path", "")).strip():
            valid["source_data_path"] = feature.pose_source_path
        valid_rows.append(valid)
        trajectories.append(feature.trajectory)
        fingertips.append(feature.fingertips)
        log_durations.append(feature.log_duration)
        if len(pose_examples) < 3:
            pose_examples.append(pose.schema_evidence)

    stem = _artifact_stem(episode_id)
    feature_path = output / "features" / "episodes" / f"{stem}.npz"
    metadata_path = output / "features" / "episodes" / f"{stem}.json"
    if feature_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Episode artifact already exists for run {run_id!r}: {feature_path}"
        )
    with feature_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            trajectory=np.asarray(trajectories, dtype=float).reshape(
                -1, core.TRAJECTORY_DIMS
            ),
            fingertips=np.asarray(fingertips, dtype=float).reshape(
                -1, core.FINGERTIP_DIMS
            ),
            log_duration=np.asarray(log_durations, dtype=float),
        )
    metadata = {
        "episode_id": str(episode_id),
        "artifact_stem": stem,
        "feature_path": str(feature_path),
        "metadata_path": str(metadata_path),
        "valid_rows": valid_rows,
        "rejected_rows": rejected_rows,
        "valid_count": len(valid_rows),
        "rejected_count": len(rejected_rows),
        "pose_examples": pose_examples,
        "segment_schema": vars(schema),
    }
    _write_json_exclusive(metadata_path, metadata)
    models_volume.commit()
    return {
        "episode_id": str(episode_id),
        "artifact_stem": stem,
        "feature_path": str(feature_path),
        "metadata_path": str(metadata_path),
        "valid_count": len(valid_rows),
        "rejected_count": len(rejected_rows),
    }


@app.function(**{**FUNCTION_OPTIONS, "memory": 8192, "timeout": 60 * 60 * 3})
def aggregate_features(
    run_id: str,
    extraction_results: list[dict[str, Any]],
    segments: str,
    pose_root: str,
    budget_frac: float,
    seed: int,
    baseline_runs: int,
) -> dict[str, Any]:
    """Combine episode artifacts and run clustering, scoring, and curation."""

    import numpy as np
    import pandas as pd

    models_volume.reload()
    output = _run_path(run_id)
    if not output.is_dir():
        raise RuntimeError(f"Run was not initialized: {output}")
    if not extraction_results:
        raise RuntimeError("No episode extraction results were supplied")
    reserved_outputs = [
        output / "features" / "valid_metadata.csv",
        output / "features" / "per_episode_index.json",
        output / "segment_scores.csv",
        output / "episode_scores.csv",
        output / "cluster_summary.csv",
        output / "nearest_neighbors.csv",
        output / "subset_manifest.csv",
        output / "subset_manifest.json",
        output / "metrics.json",
        output / "dashboard_data.json",
        output / "schema_report.json",
        output / "models",
        output / "charts",
    ]
    collisions = [str(path) for path in reserved_outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Aggregation refuses to overwrite existing run artifacts: "
            + ", ".join(collisions)
        )
    core = _core()
    trajectories: list[Any] = []
    fingertips: list[Any] = []
    log_durations: list[Any] = []
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    pose_examples: list[dict[str, Any]] = []

    for result in sorted(extraction_results, key=lambda item: str(item["episode_id"])):
        metadata_path = Path(result["metadata_path"])
        feature_path = Path(result["feature_path"])
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        with np.load(feature_path, allow_pickle=False) as payload:
            trajectories.append(np.asarray(payload["trajectory"], dtype=float))
            fingertips.append(np.asarray(payload["fingertips"], dtype=float))
            log_durations.append(np.asarray(payload["log_duration"], dtype=float))
        valid_rows.extend(metadata["valid_rows"])
        rejected_rows.extend(metadata["rejected_rows"])
        pose_examples.extend(metadata.get("pose_examples", []))

    if len(valid_rows) < 2:
        raise RuntimeError(
            f"Only {len(valid_rows)} valid segment(s) remained across extracted episodes; "
            "at least two are required for PCA"
        )
    valid_metadata = pd.DataFrame(valid_rows)
    with (output / "features" / "valid_metadata.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        valid_metadata.to_csv(stream, index=False)
    index_manifest = {
        "run_id": run_id,
        "episodes": extraction_results,
        "valid_segment_count": len(valid_rows),
        "rejected_segment_count": len(rejected_rows),
    }
    _write_json_exclusive(output / "features" / "per_episode_index.json", index_manifest)
    with (output / "modal_run_manifest.json").open("r", encoding="utf-8") as stream:
        modal_run_manifest = json.load(stream)
    schema_report = {
        "status": "mapped",
        "execution": "Modal parallel per-episode extraction",
        "segments_path": segments,
        "pose_root": pose_root,
        "valid_segment_count": len(valid_rows),
        "rejected_segment_count": len(rejected_rows),
        "pose_examples": pose_examples[:5],
        "source_mount": modal_run_manifest.get("source_volume", "read-only source mount"),
        "input_adapter": modal_run_manifest.get("input_adapter"),
        "output_path": str(output),
    }
    config = core.PipelineConfig(
        segments=Path(segments),
        pose_root=Path(pose_root),
        output_dir=output,
        budget_frac=float(budget_frac),
        seed=int(seed),
        baseline_runs=int(baseline_runs),
        enforce_local_isolation=False,
    )
    metrics = core.run_pipeline_from_features(
        valid_metadata,
        np.concatenate(trajectories, axis=0),
        np.concatenate(fingertips, axis=0),
        np.concatenate(log_durations, axis=0),
        rejected_rows,
        schema_report,
        config,
    )
    models_volume.commit()
    return {
        "run_id": run_id,
        "output_path": str(output),
        "valid_cycles": metrics["data_sanity"]["number_of_valid_cycles"],
        "rejected_cycles": metrics["data_sanity"]["number_of_rejected_cycles"],
        "selected_cycles": metrics["curation"]["selected_count"],
    }


@app.function(**FUNCTION_OPTIONS, max_containers=64, memory=4096)
def baseline_job(run_id: str, budget_frac: float, seed: int) -> dict[str, Any]:
    """Evaluate every baseline for one deterministic budget/seed pair."""

    import numpy as np
    import pandas as pd

    if not 0.0 < budget_frac <= 1.0:
        raise ValueError("budget_frac must be in (0, 1]")
    models_volume.reload()
    output = _run_path(run_id)
    embedding = np.load(output / "models" / "combined_embeddings.npy", allow_pickle=False)
    metadata = pd.read_csv(output / "features" / "valid_metadata.csv")
    scores = pd.read_csv(output / "segment_scores.csv")
    valid_scores = scores.loc[scores["is_valid"].astype(str).str.lower().eq("true")]
    clusters = valid_scores[["episode_id", "segment_id", "cluster_id"]].copy()
    metadata = metadata.merge(
        clusters,
        on=["episode_id", "segment_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if metadata["cluster_id"].isna().any():
        raise RuntimeError("Could not align aggregate cluster IDs with persisted feature rows")
    if len(metadata) != len(embedding):
        raise RuntimeError(
            f"Metadata/embedding length mismatch: {len(metadata)} versus {len(embedding)}"
        )
    cluster_ids = metadata["cluster_id"].astype(str).to_numpy()
    count_budget = max(1, int(math.floor(budget_frac * len(metadata))))
    duration_budget = budget_frac * float(metadata["duration"].sum())
    core = _core()
    summary, runs = core.run_baselines(
        embedding,
        metadata,
        cluster_ids,
        count_budget,
        duration_budget,
        int(seed),
        1,
    )
    budget_key = f"{budget_frac:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    path = output / "baselines" / "jobs" / f"budget_{budget_key}_seed_{seed}.json"
    payload = {
        "run_id": run_id,
        "budget_frac": float(budget_frac),
        "seed": int(seed),
        "count_budget": count_budget,
        "duration_budget": duration_budget,
        "summary": summary,
        "runs": runs,
    }
    _write_json_exclusive(path, payload)
    models_volume.commit()
    return {"path": str(path), **payload}


@app.function(**FUNCTION_OPTIONS)
def aggregate_baseline_jobs(
    run_id: str, job_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Persist mean/std baseline metrics across parallel deterministic jobs."""

    import numpy as np

    models_volume.reload()
    output = _run_path(run_id)
    buckets: dict[tuple[float, str], list[dict[str, float]]] = {}
    for job in job_results:
        budget = float(job["budget_frac"])
        for run in job["runs"]:
            key = (budget, str(run["baseline"]))
            buckets.setdefault(key, []).append(
                {
                    "facility_coverage": float(run["facility_coverage"]),
                    "verb_coverage": float(run["composition"]["verb_coverage"]),
                    "shannon_effective_verbs": float(
                        run["composition"]["shannon_effective_number_of_verbs"]
                    ),
                    **(
                        {
                            "weighted_execution_vendi": float(
                                run["execution"]["weighted_mean_execution_vendi"]
                            )
                        }
                        if run["execution"]["weighted_mean_execution_vendi"] is not None
                        else {}
                    ),
                }
            )
    summaries: list[dict[str, Any]] = []
    for (budget, baseline), rows in sorted(buckets.items()):
        record: dict[str, Any] = {
            "budget_frac": budget,
            "baseline": baseline,
            "run_count": len(rows),
        }
        for metric in (
            "facility_coverage",
            "verb_coverage",
            "shannon_effective_verbs",
            "weighted_execution_vendi",
        ):
            values = [row[metric] for row in rows if metric in row]
            record[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
            }
        summaries.append(record)
    payload = {
        "run_id": run_id,
        "job_count": len(job_results),
        "budgets": sorted({float(job["budget_frac"]) for job in job_results}),
        "seeds": sorted({int(job["seed"]) for job in job_results}),
        "summaries": summaries,
    }
    _write_json_exclusive(
        output / "baselines" / "parallel_baseline_summary.json", payload
    )
    models_volume.commit()
    return payload


def _automatic_run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _baseline_budgets(primary: float, smoke: bool) -> list[float]:
    if smoke:
        return [float(primary)]
    return sorted({0.20, 0.40, 0.60, float(primary)})


@app.local_entrypoint()
def main(
    segments: str = "",
    pose_root: str = "",
    egoverse_clips_root: str = "",
    egoverse_episodes_root: str = "",
    canonical_verb: str = "",
    smoke: bool = False,
    run_id: str = "",
    budget_frac: float = 0.40,
    seed: int = 42,
    baseline_runs: int = 10,
) -> None:
    """Submit extraction, aggregation, and baseline work to Modal."""

    if baseline_runs < 1:
        raise ValueError("baseline_runs must be at least 1")
    resolved_run_id = run_id.strip() or _automatic_run_id()
    use_egoverse_adapter = bool(
        egoverse_clips_root.strip() or egoverse_episodes_root.strip()
    )
    if use_egoverse_adapter:
        if segments.strip() or pose_root.strip():
            raise ValueError(
                "Use either --segments/--pose-root or the EgoVerse clip adapter, not both"
            )
        if not egoverse_clips_root.strip() or not egoverse_episodes_root.strip():
            raise ValueError(
                "--egoverse-clips-root and --egoverse-episodes-root are both required"
            )
        manifest = initialize_egoverse_run.remote(
            resolved_run_id,
            egoverse_clips_root,
            egoverse_episodes_root,
            canonical_verb,
            smoke,
            budget_frac,
            seed,
        )
        segments = str(manifest["segments"])
        pose_root = str(manifest["pose_root"])
    else:
        if not segments.strip() or not pose_root.strip():
            raise ValueError(
                "--segments and --pose-root are required unless the EgoVerse clip adapter is used"
            )
        manifest = initialize_run.remote(
            resolved_run_id,
            segments,
            pose_root,
            smoke,
            budget_frac,
            seed,
        )
    episode_ids = manifest["episode_ids"]
    extraction_tasks = [
        (resolved_run_id, episode_id, segments, pose_root, 2, "error")
        for episode_id in episode_ids
    ]
    extraction_results = list(
        extract_episode_features.starmap(
            extraction_tasks,
            order_outputs=True,
            return_exceptions=False,
        )
    )
    aggregation = aggregate_features.remote(
        resolved_run_id,
        extraction_results,
        segments,
        pose_root,
        budget_frac,
        seed,
        2 if smoke else baseline_runs,
    )

    budgets = _baseline_budgets(budget_frac, smoke)
    job_seed_count = min(2, baseline_runs) if smoke else baseline_runs
    baseline_tasks = [
        (resolved_run_id, budget, seed + offset)
        for budget in budgets
        for offset in range(job_seed_count)
    ]
    baseline_results = list(
        baseline_job.starmap(
            baseline_tasks,
            order_outputs=True,
            return_exceptions=False,
        )
    )
    baseline_summary = aggregate_baseline_jobs.remote(
        resolved_run_id, baseline_results
    )
    print(
        json.dumps(
            {
                **aggregation,
                "smoke": smoke,
                "episodes_processed": len(episode_ids),
                "parallel_baseline_jobs": baseline_summary["job_count"],
                "list_outputs_command": (
                    "python -m modal volume ls egotrim-models "
                    f"/egotrim-diversity-v2/{resolved_run_id}/"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
