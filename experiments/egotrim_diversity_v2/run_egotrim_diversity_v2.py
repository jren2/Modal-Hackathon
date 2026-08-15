#!/usr/bin/env python3
"""Isolated EgoTrim V2 diversity scoring and curation pipeline.

The module intentionally does not import repository-local helpers: this checkout had
none when the experiment was created. Real-data loading is conservative. Ambiguous
segment or pose schemas produce ``schema_report.json`` and a non-zero exit rather
than guessed mappings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
N_STEPS = 30
WRIST_INDEX = 0
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
TRAJECTORY_DIMS = N_STEPS * 3 * 2
FINGERTIP_DIMS = N_STEPS * 5 * 3 * 2
MIN_DURATION_SECONDS = 0.5
MIN_TRACKING_VALID_FRAC = 0.7
MIN_VENDI_SAMPLES = 3
MODEL_VERSION = "egotrim-diversity-v2"


class SchemaMappingError(RuntimeError):
    """Raised when real input columns or pose arrays cannot be mapped safely."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = dict(report or {})


class TrackingError(ValueError):
    """Raised when a segment has too much or unsafe missing tracking."""


@dataclass(frozen=True)
class SegmentSchema:
    episode_id: str
    segment_id: str
    verb: str
    start_time: str
    end_time: str
    duration: str | None
    tracking_valid_frac: str
    pose_ref: str | None
    video_path: str | None
    source_data: str | None
    start_frame: str | None
    end_frame_exclusive: str | None


@dataclass
class PoseSegment:
    timestamps: np.ndarray
    joints: np.ndarray  # [time, hand(left/right), joint, xyz]
    wrists: np.ndarray | None  # [time, hand(left/right), xyz], when separately provided
    coordinate_mode: str
    source_path: str
    schema_evidence: dict[str, Any]


@dataclass
class FeatureRecord:
    trajectory: np.ndarray
    fingertips: np.ndarray
    log_duration: float
    measured_tracking_valid_frac: float
    missing_value_rate: float
    interpolated_value_count: int
    coordinate_mode: str
    pose_source_path: str


@dataclass
class PipelineConfig:
    segments: Path
    pose_root: Path
    output_dir: Path
    budget_frac: float = 0.40
    seed: int = 42
    max_interp_gap: int = 2
    pca_components: int = 15
    pca_variance_target: float = 0.99
    neighbor_k: int = 5
    baseline_runs: int = 10
    clusterer: str = "kmeans"
    coordinate_fallback: str = "error"
    leave_one_out_vendi: bool = False
    enforce_local_isolation: bool = True


SEGMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "episode_id": ("episode_id", "episode", "episode_uid", "video_uid"),
    "segment_id": ("segment_id", "segment", "cycle_id", "action_cycle_id", "clip_uid"),
    "verb": ("canonical_verb", "verb", "action_verb", "lemma"),
    "start_time": ("start_time", "start_sec", "start_seconds", "start_timestamp", "t_start"),
    "end_time": ("end_time", "end_sec", "end_seconds", "end_timestamp", "t_end"),
    "duration": ("duration", "duration_sec", "duration_seconds"),
    "start_frame": ("start_frame", "frame_start", "start_frame_index"),
    "end_frame_exclusive": (
        "end_frame_exclusive",
        "end_frame",
        "frame_end_exclusive",
    ),
    "tracking_valid_frac": (
        "tracking_valid_frac",
        "tracking_valid_fraction",
        "valid_tracking_frac",
    ),
    "pose_ref": ("pose_ref", "pose_path", "pose_file", "zarr_path", "pose_identifier"),
    "video_path": ("video_path", "clip_path", "source_video_path"),
    "source_data": ("source_data_path", "source_path", "source_identifier", "data_path"),
}


POSE_ARRAY_ALIASES: dict[str, tuple[str, ...]] = {
    "combined_hands": (
        "hand_joints",
        "hand_joint_positions",
        "hands_3d",
        "joints_3d",
        "hand_pose",
    ),
    "left_hand": ("left_hand_joints", "left_hand_3d", "left_joints_3d", "left_hand_pose"),
    "right_hand": ("right_hand_joints", "right_hand_3d", "right_joints_3d", "right_hand_pose"),
    "timestamps": ("timestamps", "timestamp", "frame_timestamps", "time_seconds", "times"),
    "world_to_head": ("world_to_head", "world_T_head", "T_world_head"),
    "head_to_world": ("head_to_world", "head_T_world", "T_head_world", "head_pose"),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def ensure_output_isolated(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Local output directory must be inside isolated experiment root {EXPERIMENT_ROOT}; "
            f"received {output_dir}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_segments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Segments input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("segments", "data", "records"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise SchemaMappingError("JSON segments input must contain a record list")
        return pd.DataFrame(payload)
    raise SchemaMappingError(
        f"Unsupported segment format {suffix!r}; use CSV, Parquet, JSON, or JSONL"
    )


def _resolve_column(columns: Sequence[str], field: str, required: bool) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in columns}
    matches = [lower_to_original[a.lower()] for a in SEGMENT_ALIASES[field] if a.lower() in lower_to_original]
    if field in lower_to_original:
        return lower_to_original[field]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SchemaMappingError(
            f"Ambiguous columns for {field}: {matches}. Rename one to {field!r} or remove duplicates.",
            {"field": field, "matches": matches, "columns": list(columns)},
        )
    if required:
        raise SchemaMappingError(
            f"Missing required segment field {field!r}. Accepted names: {SEGMENT_ALIASES[field]}",
            {"field": field, "accepted_names": SEGMENT_ALIASES[field], "columns": list(columns)},
        )
    return None


def discover_segment_schema(df: pd.DataFrame) -> SegmentSchema:
    columns = [str(c) for c in df.columns]
    return SegmentSchema(
        episode_id=_resolve_column(columns, "episode_id", True) or "",
        segment_id=_resolve_column(columns, "segment_id", True) or "",
        verb=_resolve_column(columns, "verb", True) or "",
        start_time=_resolve_column(columns, "start_time", True) or "",
        end_time=_resolve_column(columns, "end_time", True) or "",
        duration=_resolve_column(columns, "duration", False),
        tracking_valid_frac=_resolve_column(columns, "tracking_valid_frac", True) or "",
        pose_ref=_resolve_column(columns, "pose_ref", False),
        video_path=_resolve_column(columns, "video_path", False),
        source_data=_resolve_column(columns, "source_data", False),
        start_frame=_resolve_column(columns, "start_frame", False),
        end_frame_exclusive=_resolve_column(columns, "end_frame_exclusive", False),
    )


def canonicalize_segments(df: pd.DataFrame, schema: SegmentSchema) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "episode_id": df[schema.episode_id].astype(str),
            "segment_id": df[schema.segment_id].astype(str),
            "canonical_verb": df[schema.verb].astype(str).str.strip().str.lower(),
            "start_time": pd.to_numeric(df[schema.start_time], errors="coerce"),
            "end_time": pd.to_numeric(df[schema.end_time], errors="coerce"),
            "tracking_valid_frac": pd.to_numeric(df[schema.tracking_valid_frac], errors="coerce"),
        }
    )
    if schema.duration:
        result["duration"] = pd.to_numeric(df[schema.duration], errors="coerce")
    else:
        result["duration"] = result["end_time"] - result["start_time"]
    if bool(schema.start_frame) != bool(schema.end_frame_exclusive):
        raise SchemaMappingError(
            "Frame slicing requires both start_frame and end_frame_exclusive columns.",
            {
                "start_frame": schema.start_frame,
                "end_frame_exclusive": schema.end_frame_exclusive,
            },
        )
    if schema.start_frame and schema.end_frame_exclusive:
        start_frame = pd.to_numeric(df[schema.start_frame], errors="coerce")
        end_frame = pd.to_numeric(df[schema.end_frame_exclusive], errors="coerce")
        invalid_frames = (
            start_frame.isna()
            | end_frame.isna()
            | (start_frame < 0)
            | (end_frame <= start_frame)
            | (start_frame % 1 != 0)
            | (end_frame % 1 != 0)
        )
        if invalid_frames.any():
            examples = result.loc[invalid_frames, "segment_id"].head(10).tolist()
            raise SchemaMappingError(
                f"Invalid frame bounds for segments: {examples}",
                {"invalid_frame_bound_segment_ids": examples},
            )
        result["start_frame"] = start_frame.astype(int)
        result["end_frame_exclusive"] = end_frame.astype(int)
    for canonical, source in (
        ("pose_ref", schema.pose_ref),
        ("video_path", schema.video_path),
        ("source_data_path", schema.source_data),
    ):
        result[canonical] = df[source].astype(str) if source else ""
    duplicate_ids = result["segment_id"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = result.loc[duplicate_ids, "segment_id"].head(10).tolist()
        raise SchemaMappingError(
            f"segment_id must be unique; duplicate examples: {examples}",
            {"duplicate_segment_ids": examples},
        )
    bad_numeric = result[["start_time", "end_time", "duration", "tracking_valid_frac"]].isna().any(axis=1)
    if bad_numeric.any():
        examples = result.loc[bad_numeric, "segment_id"].head(10).tolist()
        raise SchemaMappingError(
            f"Non-numeric or missing timestamps/duration/tracking fraction for segments: {examples}",
            {"invalid_numeric_segment_ids": examples},
        )
    return result.sort_values(["episode_id", "start_time", "segment_id"], kind="stable").reset_index(drop=True)


def _flatten_zarr_arrays(group: Any, prefix: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    try:
        attrs.update(dict(group.attrs))
    except Exception:
        pass
    for name in group.keys():
        obj = group[name]
        key = f"{prefix}/{name}" if prefix else str(name)
        normalized_key = key.replace(".", "/")
        if normalized_key.startswith("images/") or normalized_key == "annotations":
            continue
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            arrays[key] = np.asarray(obj)
        elif hasattr(obj, "keys"):
            nested_arrays, nested_attrs = _flatten_zarr_arrays(obj, key)
            arrays.update(nested_arrays)
            attrs.update({f"{key}:{k}": v for k, v in nested_attrs.items()})
    return arrays, attrs


def _load_pose_container(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if path.is_file() and path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            arrays = {k: np.asarray(payload[k]) for k in payload.files}
        attrs: dict[str, Any] = {}
        for key in ("coordinate_frame", "transform_convention"):
            if key in arrays and arrays[key].size == 1:
                attrs[key] = str(arrays.pop(key).reshape(-1)[0])
        return arrays, attrs
    if path.is_file() and path.suffix.lower() == ".npy":
        return {"hand_joints": np.load(path, allow_pickle=False)}, {}
    if path.is_dir() or path.suffix.lower() in {".zarr", ".zip"}:
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError(
                "Reading Zarr poses requires zarr; install requirements_egotrim_v2.txt"
            ) from exc
        try:
            group = zarr.open_group(str(path), mode="r")
        except Exception as exc:
            raise SchemaMappingError(f"Could not open pose Zarr {path}: {exc}") from exc
        return _flatten_zarr_arrays(group)
    raise SchemaMappingError(f"Unsupported pose container: {path}")


def _match_array(arrays: Mapping[str, np.ndarray], field: str) -> tuple[str | None, np.ndarray | None]:
    aliases = {a.lower() for a in POSE_ARRAY_ALIASES[field]}
    matches = [(k, v) for k, v in arrays.items() if k.split("/")[-1].lower() in aliases]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact = [(k, v) for k, v in matches if k.lower() == field]
        if len(exact) == 1:
            return exact[0]
        raise SchemaMappingError(
            f"Ambiguous pose arrays for {field}: {[k for k, _ in matches]}",
            {"pose_field": field, "matches": [k for k, _ in matches], "arrays": list(arrays)},
        )
    return None, None


def _normalize_hand_array(array: np.ndarray, source_name: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.ndim != 4:
        raise SchemaMappingError(
            f"Combined hand array {source_name!r} must be rank 4; found {value.shape}"
        )
    if value.shape[1] == 2 and value.shape[-1] == 3:
        result = value
    elif value.shape[-1] == 2 and value.shape[-2] == 3:
        result = np.transpose(value, (0, 3, 1, 2))
    elif value.shape[2] == 2 and value.shape[-1] == 3:
        result = np.transpose(value, (0, 2, 1, 3))
    else:
        raise SchemaMappingError(
            f"Cannot identify [time, hand, joint, xyz] axes for {source_name!r}: {value.shape}"
        )
    if result.shape[2] <= max(FINGERTIP_INDICES):
        raise SchemaMappingError(
            f"Pose array {source_name!r} has {result.shape[2]} joints; at least 21 are required"
        )
    return result


def _find_exact_path(arrays: Mapping[str, np.ndarray], *names: str) -> tuple[str | None, np.ndarray | None]:
    normalized = {key.replace(".", "/").lower(): key for key in arrays}
    for name in names:
        key = normalized.get(name.replace(".", "/").lower())
        if key is not None:
            return key, arrays[key]
    return None, None


def _quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(norms > 1e-12, norms, np.nan)
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty((*q.shape[:-1], 3, 3), dtype=float)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _pose7_head_to_world_matrices(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise SchemaMappingError(f"Expected [time,7] head pose, found {pose.shape}")
    transforms = np.zeros((len(pose), 4, 4), dtype=float)
    transforms[:, :3, :3] = _quaternion_wxyz_to_matrix(pose[:, 3:7])
    transforms[:, :3, 3] = pose[:, :3]
    transforms[:, 3, 3] = 1.0
    return transforms


def _resolve_pose_path(row: Mapping[str, Any], pose_root: Path) -> Path:
    pose_ref = str(row.get("pose_ref", "")).strip()
    if pose_ref and pose_ref.lower() not in {"nan", "none"}:
        candidate = Path(pose_ref)
        if not candidate.is_absolute():
            candidate = pose_root / candidate
        if not candidate.exists():
            raise SchemaMappingError(
                f"pose_ref for segment {row['segment_id']} does not exist: {candidate}"
            )
        return candidate
    episode = str(row["episode_id"])
    candidates: list[Path] = []
    for name in (episode, f"{episode}.zarr", f"{episode}.npz", f"{episode}.npy"):
        candidate = pose_root / name
        if candidate.exists():
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    raise SchemaMappingError(
        f"Cannot resolve pose data for episode {episode!r}; found {len(candidates)} candidates. "
        "Provide an unambiguous pose_ref column.",
        {"episode_id": episode, "pose_root": str(pose_root), "candidates": [str(p) for p in candidates]},
    )


def _apply_homogeneous(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    if transforms.shape[-2:] != (4, 4) or transforms.shape[0] != points.shape[0]:
        raise SchemaMappingError(
            f"Head transforms must have shape [time,4,4]; found {transforms.shape}"
        )
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    return np.einsum("tij,thkj->thki", transforms, homogeneous)[..., :3]


def load_pose_segment(
    row: Mapping[str, Any], pose_root: Path, coordinate_fallback: str = "error"
) -> PoseSegment:
    path = _resolve_pose_path(row, pose_root)
    arrays, attrs = _load_pose_container(path)
    combined_name, combined = _match_array(arrays, "combined_hands")
    evidence: dict[str, Any] = {
        "pose_path": str(path),
        "available_arrays": {k: list(v.shape) for k, v in arrays.items()},
        "attrs": {str(k): str(v) for k, v in attrs.items()},
    }
    egoverse_left_name, egoverse_left = _find_exact_path(arrays, "left/obs_keypoints")
    egoverse_right_name, egoverse_right = _find_exact_path(arrays, "right/obs_keypoints")
    wrists: np.ndarray | None = None
    if egoverse_left is not None and egoverse_right is not None:
        left = np.asarray(egoverse_left, dtype=float).reshape(len(egoverse_left), 21, 3)
        right = np.asarray(egoverse_right, dtype=float).reshape(len(egoverse_right), 21, 3)
        joints = np.stack([left, right], axis=1)
        lw_name, left_wrist = _find_exact_path(arrays, "left/obs_wrist_pose")
        rw_name, right_wrist = _find_exact_path(arrays, "right/obs_wrist_pose")
        if left_wrist is None or right_wrist is None:
            raise SchemaMappingError("EgoVerse schema has keypoints but is missing left/right wrist poses", evidence)
        wrists = np.stack(
            [np.asarray(left_wrist, dtype=float)[:, :3], np.asarray(right_wrist, dtype=float)[:, :3]],
            axis=1,
        )
        evidence["joint_mapping"] = {
            "left": egoverse_left_name,
            "right": egoverse_right_name,
            "left_wrist": lw_name,
            "right_wrist": rw_name,
            "joint_order": "canonical_MANO",
        }
    elif combined is not None:
        joints = _normalize_hand_array(combined, combined_name or "combined_hands")
        evidence["joint_mapping"] = combined_name
    else:
        left_name, left = _match_array(arrays, "left_hand")
        right_name, right = _match_array(arrays, "right_hand")
        if left is None or right is None:
            raise SchemaMappingError(
                "Pose data must expose one combined two-hand array or unambiguous left/right arrays.",
                evidence,
            )
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
            raise SchemaMappingError(
                f"Left/right arrays must share [time,joint,xyz] shape; found {left.shape}/{right.shape}",
                evidence,
            )
        joints = np.stack([left, right], axis=1)
        evidence["joint_mapping"] = {"left": left_name, "right": right_name}

    timestamps_name, timestamps = _find_exact_path(arrays, "obs_rgb_timestamps_ns")
    if timestamps is None:
        timestamps_name, timestamps = _match_array(arrays, "timestamps")
    frame_indices: np.ndarray | None = None
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
        total_frames = int(float(attrs.get("total_frames", len(joints))))
        total_frames = min(total_frames, len(joints), len(timestamps))
        joints = joints[:total_frames]
        if wrists is not None:
            wrists = wrists[:total_frames]
        timestamps = timestamps[:total_frames]
        if timestamps_name and timestamps_name.endswith("_ns"):
            timestamps = (timestamps - timestamps[0]) / 1e9
            evidence["timestamp_units"] = "nanoseconds_normalized_to_episode_seconds"
        if len(timestamps) != len(joints) or np.any(np.diff(timestamps) <= 0):
            raise SchemaMappingError(
                f"Timestamps {timestamps_name!r} must be strictly increasing and match pose frames",
                evidence,
            )
        has_start_frame = "start_frame" in row and not pd.isna(row["start_frame"])
        has_end_frame = "end_frame_exclusive" in row and not pd.isna(
            row["end_frame_exclusive"]
        )
        if has_start_frame != has_end_frame:
            raise SchemaMappingError(
                "A segment row must provide both start_frame and end_frame_exclusive",
                evidence,
            )
        if has_start_frame:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame_exclusive"])
            if start_frame < 0 or end_frame <= start_frame or end_frame > total_frames:
                raise TrackingError(
                    f"frame bounds [{start_frame}, {end_frame}) are outside "
                    f"the {total_frames}-frame pose store"
                )
            frame_indices = np.arange(start_frame, end_frame, dtype=int)
            evidence["frame_mapping"] = (
                f"exact_manifest_frame_bounds:{start_frame}:{end_frame}"
            )
        else:
            mask = (timestamps >= float(row["start_time"])) & (
                timestamps <= float(row["end_time"])
            )
            frame_indices = np.flatnonzero(mask)
            if len(frame_indices) < 2:
                raise TrackingError("fewer than two pose frames overlap segment timestamps")
            evidence["timestamp_mapping"] = timestamps_name
        joints = joints[frame_indices]
        if wrists is not None:
            wrists = wrists[frame_indices]
        timestamps = timestamps[frame_indices]
        if has_start_frame:
            source_deltas = np.diff(timestamps)
            evidence["source_timestamp_delta_seconds"] = {
                "median": float(np.median(source_deltas)),
                "maximum": float(np.max(source_deltas)),
            }
            timestamps = np.linspace(
                float(row["start_time"]),
                float(row["end_time"]),
                len(frame_indices),
                endpoint=False,
            )
            evidence["timestamp_mapping"] = (
                "nominal_clip_timeline_from_exact_manifest_frame_bounds"
            )
    else:
        if str(row["segment_id"]) not in path.name:
            raise SchemaMappingError(
                "Episode-level pose data has no timestamps and cannot be sliced safely. "
                "Provide timestamps or segment-specific pose_ref files.",
                evidence,
            )
        timestamps = np.linspace(float(row["start_time"]), float(row["end_time"]), len(joints))
        evidence["timestamp_mapping"] = "segment_specific_uniform_fallback"

    world_name, world_to_head = _match_array(arrays, "world_to_head")
    head_name, head_to_world = _find_exact_path(arrays, "obs_head_pose")
    if head_to_world is None:
        head_name, head_to_world = _match_array(arrays, "head_to_world")
    if world_to_head is not None and head_to_world is not None:
        raise SchemaMappingError(
            f"Both {world_name!r} and {head_name!r} are present; transform convention is ambiguous",
            evidence,
        )
    coordinate_frame = str(attrs.get("coordinate_frame", "")).lower()
    if coordinate_frame in {"head", "head_frame", "egocentric_head"}:
        coordinate_mode = "source_declared_head_frame"
    elif world_to_head is not None:
        transforms = np.asarray(world_to_head, dtype=float)
        if frame_indices is not None:
            transforms = transforms[:total_frames][frame_indices]
        joints = _apply_homogeneous(joints, transforms)
        if wrists is not None:
            wrists = _apply_homogeneous(wrists[:, :, None, :], transforms)[:, :, 0, :]
        coordinate_mode = f"per_frame_world_to_head:{world_name}"
    elif head_to_world is not None:
        transforms = np.asarray(head_to_world, dtype=float)
        if transforms.ndim == 2 and transforms.shape[1] == 7:
            transforms = _pose7_head_to_world_matrices(transforms)
            evidence["head_pose_layout"] = "xyz_qw_qx_qy_qz_from_EgoVerse_converter_convention"
        if frame_indices is not None:
            transforms = transforms[:total_frames][frame_indices]
        try:
            inverse = np.linalg.inv(transforms)
        except np.linalg.LinAlgError as exc:
            raise TrackingError("non-invertible per-frame head pose") from exc
        joints = _apply_homogeneous(joints, inverse)
        if wrists is not None:
            wrists = _apply_homogeneous(wrists[:, :, None, :], inverse)[:, :, 0, :]
        coordinate_mode = f"per_frame_inverse_head_to_world:{head_name}"
    elif coordinate_fallback == "already_head_frame":
        coordinate_mode = "explicit_cli_fallback_already_head_frame"
    else:
        raise SchemaMappingError(
            "No per-frame head transform or declared head-frame coordinates were found. This empty "
            "repository has no existing coordinate convention to reuse. Re-run only if verified with "
            "--coordinate-fallback already_head_frame, or provide a supported transform array.",
            evidence,
        )
    evidence["coordinate_mode"] = coordinate_mode
    return PoseSegment(timestamps, joints, wrists, coordinate_mode, str(path), evidence)


def interpolate_short_internal_gaps(values: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Interpolate bounded internal NaN runs; never fills edges or long gaps."""
    result = np.asarray(values, dtype=float).copy()
    original_shape = result.shape
    flat = result.reshape(len(result), -1)
    interpolated = 0
    for column in range(flat.shape[1]):
        vector = flat[:, column]
        missing = ~np.isfinite(vector)
        start = 0
        while start < len(vector):
            if not missing[start]:
                start += 1
                continue
            end = start
            while end + 1 < len(vector) and missing[end + 1]:
                end += 1
            length = end - start + 1
            if start > 0 and end < len(vector) - 1 and length <= max_gap:
                vector[start : end + 1] = np.linspace(vector[start - 1], vector[end + 1], length + 2)[1:-1]
                interpolated += length
            start = end + 1
    return flat.reshape(original_shape), interpolated


def extract_raw_feature(
    pose: PoseSegment,
    duration: float,
    max_interp_gap: int = 2,
    min_tracking_valid_frac: float = MIN_TRACKING_VALID_FRAC,
) -> FeatureRecord:
    wrist = pose.wrists if pose.wrists is not None else pose.joints[:, :, WRIST_INDEX, :]
    fingertips_native = pose.joints[:, :, FINGERTIP_INDICES, :]
    required = np.concatenate([wrist[:, :, None, :], fingertips_native], axis=2)
    # EgoVerse arrays use zero fill for missing/padded numeric chunks. Padding is
    # truncated via total_frames above; any remaining all-zero required joint is
    # treated as missing rather than as stationary motion.
    zero_joint = np.all(required == 0.0, axis=-1)
    required = required.copy()
    required[np.repeat(zero_joint[..., None], 3, axis=-1)] = np.nan
    initial_missing_rate = float(np.mean(~np.isfinite(required)))
    frame_valid = np.all(np.isfinite(required), axis=(1, 2, 3))
    measured_valid = float(np.mean(frame_valid))
    if measured_valid < min_tracking_valid_frac:
        raise TrackingError(
            f"measured required-joint tracking fraction {measured_valid:.3f} is below "
            f"{min_tracking_valid_frac:.3f}"
        )
    repaired, interpolated = interpolate_short_internal_gaps(required, max_interp_gap)
    if not np.all(np.isfinite(repaired)):
        remaining = int(np.sum(~np.isfinite(repaired)))
        raise TrackingError(
            f"{remaining} required pose values remain missing after bounded internal interpolation; "
            "leading, trailing, and long gaps are never zero-filled"
        )
    if len(pose.timestamps) < 2:
        raise TrackingError("at least two timestamps are required for resampling")
    target_time = np.linspace(float(pose.timestamps[0]), float(pose.timestamps[-1]), N_STEPS)
    flat = repaired.reshape(len(repaired), -1)
    resampled_flat = np.column_stack(
        [np.interp(target_time, pose.timestamps, flat[:, i]) for i in range(flat.shape[1])]
    )
    resampled = resampled_flat.reshape(N_STEPS, 2, 6, 3)
    wrist = resampled[:, :, 0, :]
    wrist_displacement = wrist - wrist[0:1]
    trajectory = np.transpose(wrist_displacement, (0, 2, 1)).reshape(-1)
    fingertips_relative = resampled[:, :, 1:, :] - wrist[:, :, None, :]
    fingertips = np.transpose(fingertips_relative, (0, 2, 3, 1)).reshape(-1)
    if trajectory.shape != (TRAJECTORY_DIMS,) or fingertips.shape != (FINGERTIP_DIMS,):
        raise AssertionError(
            f"Unexpected feature dimensions: trajectory={trajectory.shape}, fingertips={fingertips.shape}"
        )
    return FeatureRecord(
        trajectory=trajectory,
        fingertips=fingertips,
        log_duration=float(np.log(duration)),
        measured_tracking_valid_frac=measured_valid,
        missing_value_rate=initial_missing_rate,
        interpolated_value_count=interpolated,
        coordinate_mode=pose.coordinate_mode,
        pose_source_path=pose.source_path,
    )


class BalancedEmbeddingModel:
    """Independent block scaling/PCA followed by final combined scaling."""

    def __init__(self, max_components: int = 15, variance_target: float = 0.99):
        self.max_components = max_components
        self.variance_target = variance_target

    @staticmethod
    def _component_count(pca: Any, cap: int, target: float) -> int:
        ratios = np.asarray(pca.explained_variance_ratio_)
        positive = int(np.sum(np.asarray(pca.explained_variance_) > 1e-12))
        if positive == 0:
            return 1
        within = min(cap, positive)
        reached = np.flatnonzero(np.cumsum(ratios[:within]) >= target)
        # Retain close to the requested cap while dropping numerically empty dimensions.
        return max(1, min(within, int(reached[0] + 1) if len(reached) else within))

    def fit_transform(
        self, trajectory: np.ndarray, fingertips: np.ndarray, log_duration: np.ndarray
    ) -> dict[str, np.ndarray]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        trajectory = np.asarray(trajectory, dtype=float)
        fingertips = np.asarray(fingertips, dtype=float)
        log_duration = np.asarray(log_duration, dtype=float).reshape(-1, 1)
        if len(trajectory) < 2:
            raise ValueError("At least two valid segments are required to fit balanced PCA embeddings")
        self.trajectory_scaler = StandardScaler().fit(trajectory)
        self.fingertip_scaler = StandardScaler().fit(fingertips)
        traj_scaled = self.trajectory_scaler.transform(trajectory)
        tip_scaled = self.fingertip_scaler.transform(fingertips)
        cap_traj = max(1, min(self.max_components, len(trajectory) - 1, trajectory.shape[1]))
        cap_tip = max(1, min(self.max_components, len(fingertips) - 1, fingertips.shape[1]))
        traj_probe = PCA(n_components=cap_traj, svd_solver="full").fit(traj_scaled)
        tip_probe = PCA(n_components=cap_tip, svd_solver="full").fit(tip_scaled)
        self.trajectory_components_ = self._component_count(traj_probe, cap_traj, self.variance_target)
        self.fingertip_components_ = self._component_count(tip_probe, cap_tip, self.variance_target)
        self.trajectory_pca = PCA(n_components=self.trajectory_components_, svd_solver="full").fit(traj_scaled)
        self.fingertip_pca = PCA(n_components=self.fingertip_components_, svd_solver="full").fit(tip_scaled)
        traj_pca = self.trajectory_pca.transform(traj_scaled)
        tip_pca = self.fingertip_pca.transform(tip_scaled)
        self.duration_scaler = StandardScaler().fit(log_duration)
        duration_scaled = self.duration_scaler.transform(log_duration)
        self.trajectory_ablation_scaler = StandardScaler().fit(traj_pca)
        self.fingertip_ablation_scaler = StandardScaler().fit(tip_pca)
        combined_raw = np.column_stack([traj_pca, tip_pca, duration_scaled])
        self.final_scaler = StandardScaler().fit(combined_raw)
        return {
            "trajectory": self.trajectory_ablation_scaler.transform(traj_pca),
            "fingertips": self.fingertip_ablation_scaler.transform(tip_pca),
            "duration": duration_scaled,
            "combined": self.final_scaler.transform(combined_raw),
        }

    def save(self, models_dir: Path) -> None:
        import joblib

        models_dir.mkdir(parents=True, exist_ok=True)
        objects = {
            "trajectory_scaler": self.trajectory_scaler,
            "fingertip_scaler": self.fingertip_scaler,
            "trajectory_pca": self.trajectory_pca,
            "fingertip_pca": self.fingertip_pca,
            "duration_scaler": self.duration_scaler,
            "trajectory_ablation_scaler": self.trajectory_ablation_scaler,
            "fingertip_ablation_scaler": self.fingertip_ablation_scaler,
            "final_combined_scaler": self.final_scaler,
        }
        for name, model in objects.items():
            joblib.dump(model, models_dir / f"{name}.joblib")
        write_json(
            models_dir / "embedding_model_metadata.json",
            {
                "model_version": MODEL_VERSION,
                "trajectory_input_dimensions": TRAJECTORY_DIMS,
                "fingertip_input_dimensions": FINGERTIP_DIMS,
                "trajectory_components": self.trajectory_components_,
                "fingertip_components": self.fingertip_components_,
                "combined_dimensions": self.trajectory_components_ + self.fingertip_components_ + 1,
                "feature_order": {
                    "trajectory": "[time, xyz, hand(left,right)]",
                    "fingertips": "[time, fingertip(4,8,12,16,20), xyz, hand(left,right)]",
                },
                "pca_variance_target": self.variance_target,
                "pca_component_cap": self.max_components,
            },
        )


def pairwise_distances(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=float)
    squared = np.sum(embedding * embedding, axis=1, keepdims=True)
    return np.sqrt(np.maximum(squared + squared.T - 2 * embedding @ embedding.T, 0.0))


def rbf_similarity(embedding: np.ndarray) -> tuple[np.ndarray, float]:
    distances = pairwise_distances(embedding)
    nonzero = distances[distances > 1e-12]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
    kernel = np.exp(-(distances**2) / (2.0 * bandwidth**2))
    np.fill_diagonal(kernel, 1.0)
    return kernel, bandwidth


def vendi_from_kernel(kernel: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        return float("nan")
    probabilities = eigenvalues[eigenvalues > 1e-15] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def composition_metrics(verbs: Sequence[str]) -> dict[str, Any]:
    values, counts = np.unique(np.asarray(verbs, dtype=str), return_counts=True)
    probabilities = counts / counts.sum() if counts.sum() else np.array([])
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if len(probabilities) else 0.0
    return {
        "verb_coverage": int(len(values)),
        "shannon_entropy": entropy,
        "shannon_effective_number_of_verbs": float(np.exp(entropy)) if len(values) else 0.0,
        "verb_distribution": {str(v): int(c) for v, c in zip(values, counts)},
        "definition_note": "Histogram entropy is reported as Shannon effective verbs, not Vendi.",
    }


def execution_vendi_by_verb(
    embedding: np.ndarray, verbs: Sequence[str], min_samples: int = MIN_VENDI_SAMPLES
) -> dict[str, Any]:
    verbs_array = np.asarray(verbs, dtype=str)
    by_verb: dict[str, Any] = {}
    weighted_sum = 0.0
    weighted_count = 0
    for verb in sorted(np.unique(verbs_array)):
        indices = np.flatnonzero(verbs_array == verb)
        if len(indices) < min_samples:
            by_verb[verb] = {
                "sample_count": int(len(indices)),
                "status": "too_few_samples",
                "vendi": None,
                "bandwidth": None,
            }
            continue
        kernel, bandwidth = rbf_similarity(embedding[indices])
        vendi = vendi_from_kernel(kernel)
        status = "all_identical" if bandwidth == 1.0 and np.allclose(kernel, 1.0) else "ok"
        by_verb[verb] = {
            "sample_count": int(len(indices)),
            "status": status,
            "vendi": vendi,
            "bandwidth": bandwidth,
        }
        weighted_sum += len(indices) * vendi
        weighted_count += len(indices)
    return {
        "by_verb": by_verb,
        "weighted_mean_execution_vendi": weighted_sum / weighted_count if weighted_count else None,
        "weighted_sample_count": weighted_count,
        "minimum_reliable_samples": min_samples,
    }


def _canonicalize_cluster_labels(labels: np.ndarray, segment_ids: Sequence[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    mapping: dict[int, int] = {}
    next_label = 0
    representatives: list[tuple[str, int]] = []
    for label in sorted(set(labels) - {-1}):
        ids = [str(segment_ids[i]) for i in np.flatnonzero(labels == label)]
        representatives.append((min(ids), int(label)))
    for _, label in sorted(representatives):
        mapping[label] = next_label
        next_label += 1
    return np.asarray([-1 if x == -1 else mapping[int(x)] for x in labels], dtype=int)


def cluster_within_verbs(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    clusterer: str = "kmeans",
    max_k: int = 8,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, silhouette_score

    verbs = metadata["canonical_verb"].astype(str).to_numpy()
    segment_ids = metadata["segment_id"].astype(str).to_numpy()
    cluster_ids = np.empty(len(metadata), dtype=object)
    summary: list[dict[str, Any]] = []
    per_verb_selection: dict[str, Any] = {}
    for verb in sorted(np.unique(verbs)):
        indices = np.flatnonzero(verbs == verb)
        local = embedding[indices]
        if clusterer == "hdbscan":
            try:
                import hdbscan
            except ImportError as exc:
                raise RuntimeError(
                    "--clusterer hdbscan requires the optional hdbscan package; install it only in "
                    "this experiment environment"
                ) from exc
            min_cluster_size = max(2, min(10, len(indices) // 4))
            if len(indices) < min_cluster_size:
                labels = np.zeros(len(indices), dtype=int)
            else:
                labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(local)
            chosen_k: int | str = "hdbscan"
            silhouette = None
        elif len(indices) < 4 or np.allclose(pairwise_distances(local), 0.0):
            labels = np.zeros(len(indices), dtype=int)
            chosen_k = 1
            silhouette = None
        else:
            candidates: list[tuple[float, int, np.ndarray]] = []
            for k in range(2, min(max_k, len(indices) - 1) + 1):
                model = KMeans(
                    n_clusters=k,
                    random_state=(seed + stable_int(verb)) % (2**31 - 1),
                    n_init=20,
                    max_iter=300,
                    algorithm="lloyd",
                )
                candidate_labels = model.fit_predict(local)
                if len(np.unique(candidate_labels)) < 2:
                    continue
                score = float(silhouette_score(local, candidate_labels, metric="euclidean"))
                candidates.append((score, k, candidate_labels))
            if not candidates:
                labels = np.zeros(len(indices), dtype=int)
                chosen_k = 1
                silhouette = None
            else:
                silhouette, chosen_k, labels = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        labels = _canonicalize_cluster_labels(labels, segment_ids[indices])
        per_verb_selection[verb] = {
            "sample_count": int(len(indices)),
            "chosen_k": chosen_k,
            "silhouette": silhouette,
            "noise_count": int(np.sum(labels == -1)),
        }
        for label in sorted(np.unique(labels)):
            member_local = np.flatnonzero(labels == label)
            member_global = indices[member_local]
            cluster_name = f"{verb}__noise" if label == -1 else f"{verb}__{int(label)}"
            cluster_ids[member_global] = cluster_name
            if label == -1:
                representative_global = member_global[np.argmin(segment_ids[member_global])]
            else:
                distances = pairwise_distances(embedding[member_global])
                medoid_local = int(np.argmin(distances.mean(axis=1)))
                representative_global = int(member_global[medoid_local])
            episode_counts = metadata.iloc[member_global]["episode_id"].astype(str).value_counts()
            summary.append(
                {
                    "canonical_verb": verb,
                    "cluster_id": cluster_name,
                    "cluster_size": int(len(member_global)),
                    "is_noise": bool(label == -1),
                    "dominant_episode_id": str(episode_counts.index[0]),
                    "dominant_episode_count": int(episode_counts.iloc[0]),
                    "representative_segment_id": str(metadata.iloc[representative_global]["segment_id"]),
                    "representative_episode_id": str(metadata.iloc[representative_global]["episode_id"]),
                    "representative_start_time": float(metadata.iloc[representative_global]["start_time"]),
                    "silhouette_for_verb": silhouette,
                }
            )

    unique_verbs = sorted(np.unique(verbs))
    global_k = min(max(1, len(unique_verbs)), max(1, len(metadata) - 1))
    if global_k > 1:
        global_labels = KMeans(
            n_clusters=global_k,
            random_state=seed,
            n_init=20,
            max_iter=300,
            algorithm="lloyd",
        ).fit_predict(embedding)
        purity_count = 0
        for label in np.unique(global_labels):
            local_verbs = verbs[global_labels == label]
            _, counts = np.unique(local_verbs, return_counts=True)
            purity_count += int(counts.max())
        global_audit = {
            "cluster_count": global_k,
            "verb_purity": purity_count / len(metadata),
            "normalized_mutual_information_with_verb": float(
                normalized_mutual_info_score(verbs, global_labels)
            ),
            "interpretation_guardrail": (
                "High alignment indicates the global embedding recovers verbs; lower alignment can "
                "indicate cross-verb motion similarity but is not, by itself, a behavioral conclusion."
            ),
        }
    else:
        global_audit = {
            "cluster_count": 1,
            "verb_purity": 1.0,
            "normalized_mutual_information_with_verb": 1.0,
            "interpretation_guardrail": "Only one global cluster/verb was available.",
        }
    return cluster_ids.astype(str), summary, {
        "per_verb": per_verb_selection,
        "global_audit": global_audit,
    }


def distinctiveness_scores(
    embedding: np.ndarray, metadata: pd.DataFrame, cluster_ids: Sequence[str], k: int = 5
) -> pd.DataFrame:
    verbs = metadata["canonical_verb"].astype(str).to_numpy()
    tracking = metadata["tracking_quality"].astype(float).to_numpy()
    mean_knn = np.full(len(metadata), np.nan)
    percentile = np.full(len(metadata), np.nan)
    cluster_rarity = np.zeros(len(metadata), dtype=float)
    cluster_ids_array = np.asarray(cluster_ids, dtype=str)
    for verb in sorted(np.unique(verbs)):
        indices = np.flatnonzero(verbs == verb)
        if len(indices) > 1:
            distances = pairwise_distances(embedding[indices])
            np.fill_diagonal(distances, np.inf)
            local_k = min(k, len(indices) - 1)
            local_means = np.sort(distances, axis=1)[:, :local_k].mean(axis=1)
            mean_knn[indices] = local_means
            order = pd.Series(local_means).rank(method="average", pct=True).to_numpy()
            percentile[indices] = order
        for cluster_id in np.unique(cluster_ids_array[indices]):
            members = indices[cluster_ids_array[indices] == cluster_id]
            cluster_rarity[members] = 1.0 - (len(members) / len(indices))
    potential_tracking_outlier = (
        np.nan_to_num(percentile, nan=0.0) >= 0.90
    ) & (tracking < 0.85)
    return pd.DataFrame(
        {
            "mean_same_verb_knn_distance": mean_knn,
            "distinctiveness_percentile": percentile,
            "cluster_rarity": cluster_rarity,
            "potential_tracking_outlier": potential_tracking_outlier,
        }
    )


def nearest_neighbor_audit(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    representative_count: int = 20,
    neighbors_per_query: int = 5,
) -> tuple[pd.DataFrame, float | None]:
    if len(metadata) < 2:
        return pd.DataFrame(), None
    distances = pairwise_distances(embedding)
    np.fill_diagonal(distances, np.inf)
    query_count = min(max(20, representative_count), len(metadata))
    if query_count == len(metadata):
        query_indices = np.arange(len(metadata))
    else:
        query_indices = np.unique(np.linspace(0, len(metadata) - 1, query_count).round().astype(int))
    rows: list[dict[str, Any]] = []
    same_verb_first = []
    for query_index in query_indices:
        order = np.argsort(distances[query_index], kind="stable")[: min(neighbors_per_query, len(metadata) - 1)]
        query = metadata.iloc[query_index]
        for rank, neighbor_index in enumerate(order, start=1):
            neighbor = metadata.iloc[int(neighbor_index)]
            same_verb = str(query["canonical_verb"]) == str(neighbor["canonical_verb"])
            if rank == 1:
                same_verb_first.append(same_verb)
            rows.append(
                {
                    "query_segment_id": str(query["segment_id"]),
                    "query_episode_id": str(query["episode_id"]),
                    "query_verb": str(query["canonical_verb"]),
                    "query_start_time": float(query["start_time"]),
                    "query_end_time": float(query["end_time"]),
                    "query_video_path": str(query.get("video_path", "")),
                    "neighbor_rank": rank,
                    "neighbor_segment_id": str(neighbor["segment_id"]),
                    "neighbor_episode_id": str(neighbor["episode_id"]),
                    "neighbor_verb": str(neighbor["canonical_verb"]),
                    "neighbor_start_time": float(neighbor["start_time"]),
                    "neighbor_end_time": float(neighbor["end_time"]),
                    "neighbor_video_path": str(neighbor.get("video_path", "")),
                    "distance": float(distances[query_index, neighbor_index]),
                    "same_verb": same_verb,
                }
            )
    return pd.DataFrame(rows), float(np.mean(same_verb_first)) if same_verb_first else None


def _facility_similarity(embedding: np.ndarray) -> tuple[np.ndarray, float]:
    return rbf_similarity(embedding)


def facility_value(similarity: np.ndarray, selected: Sequence[int]) -> float:
    if not selected:
        return 0.0
    return float(np.max(similarity[:, list(selected)], axis=1).sum())


def select_facility_subset(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    count_budget: int,
    duration_budget: float,
    min_populated_verb: int = 3,
) -> tuple[list[int], dict[int, str], dict[str, Any]]:
    similarity, bandwidth = _facility_similarity(embedding)
    durations = metadata["duration"].astype(float).to_numpy()
    verbs = metadata["canonical_verb"].astype(str).to_numpy()
    quality = metadata["tracking_quality"].astype(float).to_numpy()
    outliers = metadata["potential_tracking_outlier"].astype(bool).to_numpy()
    eligible_verbs = [v for v in sorted(np.unique(verbs)) if int(np.sum(verbs == v)) >= min_populated_verb]
    selected: list[int] = []
    reasons: dict[int, str] = {}
    current_coverage = np.zeros(len(metadata), dtype=float)

    def feasible(index: int) -> bool:
        return (
            index not in selected
            and len(selected) < count_budget
            and float(durations[selected].sum() if selected else 0.0) + durations[index]
            <= duration_budget + 1e-9
        )

    preserved: list[str] = []
    for verb in eligible_verbs:
        candidates = [
            int(i)
            for i in np.flatnonzero(verbs == verb)
            if not outliers[i] and feasible(int(i))
        ]
        if not candidates:
            continue
        verb_indices = np.flatnonzero(verbs == verb)
        # Central, high-quality representative; stable segment ID resolves ties.
        ranked = []
        for index in candidates:
            centrality = float(similarity[verb_indices, index].mean())
            ranked.append(
                (-centrality * quality[index], str(metadata.iloc[index]["segment_id"]), index)
            )
        index = sorted(ranked)[0][2]
        selected.append(index)
        reasons[index] = "preserve_populated_verb"
        preserved.append(verb)
        current_coverage = np.maximum(current_coverage, similarity[:, index])

    while len(selected) < count_budget:
        candidates = [i for i in range(len(metadata)) if feasible(i) and not outliers[i]]
        if not candidates:
            break
        ranked = []
        for index in candidates:
            gain = float(np.maximum(current_coverage, similarity[:, index]).sum() - current_coverage.sum())
            gain_per_second = gain / max(durations[index], 1e-9)
            score = gain_per_second * max(quality[index], 0.0)
            ranked.append((-score, -gain, str(metadata.iloc[index]["segment_id"]), index))
        _, _, _, index = sorted(ranked)[0]
        selected.append(index)
        reasons[index] = "facility_location_gain_per_second"
        current_coverage = np.maximum(current_coverage, similarity[:, index])

    audit = {
        "count_budget": int(count_budget),
        "duration_budget": float(duration_budget),
        "selected_count": int(len(selected)),
        "selected_duration": float(durations[selected].sum() if selected else 0.0),
        "facility_value": float(current_coverage.sum()),
        "facility_value_normalized": float(current_coverage.mean()),
        "similarity_bandwidth": bandwidth,
        "eligible_populated_verbs": eligible_verbs,
        "preserved_populated_verbs": preserved,
        "verb_preservation_feasible": set(preserved) == set(eligible_verbs),
        "excluded_tracking_outlier_count": int(np.sum(outliers)),
        "objective": "monotonic facility location; duration-aware marginal gain with tracking guardrail",
    }
    return selected, reasons, audit


def _select_from_order(
    order: Iterable[int], durations: np.ndarray, count_budget: int, duration_budget: float
) -> list[int]:
    selected: list[int] = []
    total_duration = 0.0
    for raw_index in order:
        index = int(raw_index)
        if len(selected) >= count_budget:
            break
        if total_duration + durations[index] <= duration_budget + 1e-9:
            selected.append(index)
            total_duration += durations[index]
    return selected


def annotation_only_order(metadata: pd.DataFrame) -> list[int]:
    """Round-robin verbs, choosing median-duration examples; uses no motion fields."""
    queues: dict[str, list[int]] = {}
    for verb, group in metadata.groupby("canonical_verb", sort=True):
        median = float(group["duration"].median())
        queues[str(verb)] = sorted(
            [int(i) for i in group.index],
            key=lambda i: (abs(float(metadata.iloc[i]["duration"]) - median), str(metadata.iloc[i]["segment_id"])),
        )
    order: list[int] = []
    while any(queues.values()):
        for verb in sorted(queues):
            if queues[verb]:
                order.append(queues[verb].pop(0))
    return order


def stratified_random_order(metadata: pd.DataFrame, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    queues: dict[str, list[int]] = {}
    for verb, group in metadata.groupby("canonical_verb", sort=True):
        values = group.index.to_numpy(dtype=int, copy=True)
        rng.shuffle(values)
        queues[str(verb)] = values.tolist()
    order: list[int] = []
    while any(queues.values()):
        for verb in sorted(queues):
            if queues[verb]:
                order.append(queues[verb].pop(0))
    return order


def cluster_medoid_order(
    embedding: np.ndarray, metadata: pd.DataFrame, cluster_ids: Sequence[str]
) -> list[int]:
    cluster_ids = np.asarray(cluster_ids, dtype=str)
    medoids: list[int] = []
    remaining: list[int] = []
    for cluster_id in sorted(np.unique(cluster_ids)):
        members = np.flatnonzero(cluster_ids == cluster_id)
        local_distances = pairwise_distances(embedding[members])
        medoid = int(members[np.argmin(local_distances.mean(axis=1))])
        medoids.append(medoid)
        remaining.extend(int(i) for i in members if int(i) != medoid)
    return medoids + sorted(remaining, key=lambda i: str(metadata.iloc[i]["segment_id"]))


def evaluate_subset(
    indices: Sequence[int], embedding: np.ndarray, metadata: pd.DataFrame, full_similarity: np.ndarray
) -> dict[str, Any]:
    indices = list(map(int, indices))
    if not indices:
        return {
            "selected_count": 0,
            "selected_duration": 0.0,
            "composition": composition_metrics([]),
            "execution": {"by_verb": {}, "weighted_mean_execution_vendi": None},
            "facility_coverage": 0.0,
        }
    subset_meta = metadata.iloc[indices]
    return {
        "selected_count": len(indices),
        "selected_duration": float(subset_meta["duration"].sum()),
        "composition": composition_metrics(subset_meta["canonical_verb"].astype(str).tolist()),
        "execution": execution_vendi_by_verb(
            embedding[indices], subset_meta["canonical_verb"].astype(str).tolist()
        ),
        "facility_coverage": float(np.max(full_similarity[:, indices], axis=1).mean()),
    }


def run_baselines(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    cluster_ids: Sequence[str],
    count_budget: int,
    duration_budget: float,
    seed: int,
    runs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    durations = metadata["duration"].astype(float).to_numpy()
    similarity, _ = _facility_similarity(embedding)
    per_run: list[dict[str, Any]] = []
    fixed_orders = {
        "annotation_only": annotation_only_order(metadata),
        "cluster_medoid": cluster_medoid_order(embedding, metadata, cluster_ids),
    }
    for run_index in range(runs):
        run_seed = seed + run_index
        rng = np.random.default_rng(run_seed)
        orders = {
            "uniform_random": rng.permutation(len(metadata)).tolist(),
            "verb_stratified_random": stratified_random_order(metadata, run_seed),
            **fixed_orders,
        }
        for name, order in orders.items():
            selected = _select_from_order(order, durations, count_budget, duration_budget)
            evaluated = evaluate_subset(selected, embedding, metadata, similarity)
            per_run.append(
                {
                    "baseline": name,
                    "run_index": run_index,
                    "seed": run_seed,
                    "selected_indices": selected,
                    "selected_segment_ids": metadata.iloc[selected]["segment_id"].astype(str).tolist(),
                    **evaluated,
                }
            )
    summary: dict[str, Any] = {}
    for name in sorted({row["baseline"] for row in per_run}):
        rows = [row for row in per_run if row["baseline"] == name]
        summary[name] = {}
        for field, getter in (
            ("facility_coverage", lambda x: x["facility_coverage"]),
            ("verb_coverage", lambda x: x["composition"]["verb_coverage"]),
            (
                "shannon_effective_verbs",
                lambda x: x["composition"]["shannon_effective_number_of_verbs"],
            ),
            (
                "weighted_execution_vendi",
                lambda x: x["execution"]["weighted_mean_execution_vendi"],
            ),
        ):
            values = np.asarray([getter(row) for row in rows if getter(row) is not None], dtype=float)
            summary[name][field] = {
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=0)) if len(values) else None,
            }
        summary[name]["runs"] = runs
    return summary, per_run


def ablation_metrics(
    embeddings: Mapping[str, np.ndarray], metadata: pd.DataFrame
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("trajectory", "fingertips", "duration", "combined"):
        _, same_verb_rate = nearest_neighbor_audit(
            embeddings[name], metadata, representative_count=min(20, len(metadata)), neighbors_per_query=1
        )
        execution = execution_vendi_by_verb(
            embeddings[name], metadata["canonical_verb"].astype(str).tolist()
        )
        result[name] = {
            "dimensions": int(embeddings[name].shape[1]),
            "same_verb_nearest_neighbor_rate": same_verb_rate,
            "weighted_mean_execution_vendi": execution["weighted_mean_execution_vendi"],
        }
    return result


def episode_scores(metadata: pd.DataFrame, selected: Sequence[int]) -> pd.DataFrame:
    selected_set = set(map(int, selected))
    full_verb_count = max(1, metadata["canonical_verb"].nunique())
    rows: list[dict[str, Any]] = []
    for episode_id, group in metadata.groupby("episode_id", sort=True):
        indices = group.index.to_numpy(dtype=int)
        distinctiveness = group["distinctiveness_percentile"].dropna()
        rare_coverage = float(np.mean(group["cluster_rarity"].astype(float) >= 0.5))
        mean_distinctiveness = float(distinctiveness.mean()) if len(distinctiveness) else 0.0
        max_distinctiveness = float(distinctiveness.max()) if len(distinctiveness) else 0.0
        verb_coverage_component = group["canonical_verb"].nunique() / full_verb_count
        tracking_quality = float(group["tracking_quality"].mean())
        composite = (
            0.35 * mean_distinctiveness
            + 0.15 * max_distinctiveness
            + 0.20 * verb_coverage_component
            + 0.15 * rare_coverage
            + 0.15 * tracking_quality
        )
        rows.append(
            {
                "episode_id": str(episode_id),
                "number_valid_cycles": int(len(group)),
                "verbs_represented": "|".join(sorted(group["canonical_verb"].astype(str).unique())),
                "verb_count": int(group["canonical_verb"].nunique()),
                "mean_distinctiveness": mean_distinctiveness,
                "maximum_distinctiveness": max_distinctiveness,
                "rare_cluster_coverage": rare_coverage,
                "total_usable_duration": float(group["duration"].sum()),
                "tracking_quality": tracking_quality,
                "selected_segment_count": int(sum(i in selected_set for i in indices)),
                "episode_score": composite,
            }
        )
    result = pd.DataFrame(rows)
    result["episode_rank"] = result["episode_score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["episode_rank", "episode_id"], kind="stable").reset_index(drop=True)


def _save_chart(path: Path, title: str, draw: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    draw(ax)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_charts(
    charts_dir: Path,
    metadata: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    model: BalancedEmbeddingModel,
    execution: Mapping[str, Any],
    same_verb_rate: float | None,
    ablations: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    coverage_curve: Sequence[Mapping[str, Any]],
) -> list[str]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    verb_counts = metadata["canonical_verb"].value_counts().sort_index()
    path = charts_dir / "verb_composition.png"
    _save_chart(
        path,
        "Verb Composition of Valid Action Cycles",
        lambda ax: (
            ax.bar(verb_counts.index, verb_counts.values, color="#3A6EA5"),
            ax.set_ylabel("Action cycles"),
            ax.tick_params(axis="x", rotation=35),
        ),
    )
    paths.append(path)

    vendi_rows = [
        (verb, data["vendi"])
        for verb, data in execution["by_verb"].items()
        if data["vendi"] is not None
    ]
    path = charts_dir / "execution_diversity_by_verb.png"
    _save_chart(
        path,
        "Execution Diversity Within Each Verb",
        lambda ax: (
            ax.bar([x[0] for x in vendi_rows], [x[1] for x in vendi_rows], color="#2A9D8F"),
            ax.set_ylabel("RBF-kernel Vendi score"),
            ax.tick_params(axis="x", rotation=35),
        ),
    )
    paths.append(path)

    path = charts_dir / "cluster_sizes.png"
    _save_chart(
        path,
        "Execution-Style Cluster Sizes",
        lambda ax: (
            ax.bar(
                cluster_summary["cluster_id"].astype(str),
                cluster_summary["cluster_size"].astype(int),
                color="#E9C46A",
            ),
            ax.set_ylabel("Segments"),
            ax.tick_params(axis="x", rotation=60, labelsize=8),
        ),
    )
    paths.append(path)

    traj_variance = np.asarray(model.trajectory_pca.explained_variance_ratio_)
    tip_variance = np.asarray(model.fingertip_pca.explained_variance_ratio_)
    path = charts_dir / "pca_explained_variance.png"
    def draw_pca(ax: Any) -> None:
        ax.plot(np.arange(1, len(traj_variance) + 1), np.cumsum(traj_variance), marker="o", label="Trajectory")
        ax.plot(np.arange(1, len(tip_variance) + 1), np.cumsum(tip_variance), marker="o", label="Fingertips")
        ax.set_xlabel("Retained components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_ylim(0, 1.05)
        ax.legend()
    _save_chart(path, "Balanced PCA Explained Variance", draw_pca)
    paths.append(path)

    path = charts_dir / "same_verb_nearest_neighbor_rate.png"
    rate = float(same_verb_rate or 0.0)
    _save_chart(
        path,
        "Nearest-Neighbor Verb Agreement",
        lambda ax: (
            ax.bar(["Same verb", "Different verb"], [rate, 1.0 - rate], color=["#3A6EA5", "#B8C4D2"]),
            ax.set_ylabel("Fraction of representative queries"),
            ax.set_ylim(0, 1),
        ),
    )
    paths.append(path)

    path = charts_dir / "coverage_vs_budget.png"
    def draw_curve(ax: Any) -> None:
        for method in sorted({str(row["method"]) for row in coverage_curve}):
            points = [row for row in coverage_curve if row["method"] == method]
            ax.plot(
                [100 * float(row["budget_frac"]) for row in points],
                [float(row["facility_coverage"]) for row in points],
                marker="o",
                label=method.replace("_", " ").title(),
            )
        ax.set_xlabel("Data budget (%)")
        ax.set_ylabel("Facility-location coverage")
        ax.legend(fontsize=8)
    _save_chart(path, "Measured Coverage Across Data Budgets", draw_curve)
    paths.append(path)

    path = charts_dir / "egotrim_vs_baselines.png"
    names = ["egotrim"] + sorted(baseline_summary)
    values = [
        next((float(r["facility_coverage"]) for r in coverage_curve if r["method"] == "egotrim" and math.isclose(float(r["budget_frac"]), float(coverage_curve[0].get("primary_budget_frac", -1)))), np.nan)
    ]
    # The primary EgoTrim value is carried separately by the caller in the final row.
    egotrim_primary = next(
        (float(row["facility_coverage"]) for row in coverage_curve if row.get("is_primary") and row["method"] == "egotrim"),
        float("nan"),
    )
    values = [egotrim_primary] + [float(baseline_summary[n]["facility_coverage"]["mean"]) for n in names[1:]]
    errors = [0.0] + [float(baseline_summary[n]["facility_coverage"]["std"]) for n in names[1:]]
    _save_chart(
        path,
        "EgoTrim and Baseline Behavioral Coverage",
        lambda ax: (
            ax.bar([n.replace("_", " ").title() for n in names], values, yerr=errors, color="#3A6EA5"),
            ax.set_ylabel("Facility-location coverage"),
            ax.tick_params(axis="x", rotation=30),
        ),
    )
    paths.append(path)

    ranked = metadata.sort_values("distinctiveness_percentile", kind="stable")
    bottom = ranked.head(min(5, len(ranked)))
    top = ranked.tail(min(5, len(ranked)))
    labels = [f"Bottom: {x}" for x in bottom["segment_id"].astype(str)] + [
        f"Top: {x}" for x in top["segment_id"].astype(str)
    ]
    scores = bottom["distinctiveness_percentile"].astype(float).tolist() + top[
        "distinctiveness_percentile"
    ].astype(float).tolist()
    path = charts_dir / "top_bottom_ranked_segments.png"
    _save_chart(
        path,
        "Within-Verb Distinctiveness: Top and Bottom Segments",
        lambda ax: (
            ax.barh(labels, scores, color=["#B8C4D2"] * len(bottom) + ["#E76F51"] * len(top)),
            ax.set_xlabel("Within-verb distinctiveness percentile"),
            ax.set_xlim(0, 1),
        ),
    )
    paths.append(path)

    path = charts_dir / "feature_ablation_comparison.png"
    ablation_names = list(ablations)
    _save_chart(
        path,
        "Feature Ablation: Nearest-Neighbor Verb Agreement",
        lambda ax: (
            ax.bar(
                [x.title() for x in ablation_names],
                [float(ablations[x]["same_verb_nearest_neighbor_rate"] or 0.0) for x in ablation_names],
                color="#8E7DBE",
            ),
            ax.set_ylabel("Same-verb nearest-neighbor rate"),
            ax.set_ylim(0, 1),
        ),
    )
    paths.append(path)
    return [str(path.relative_to(charts_dir.parent)).replace("\\", "/") for path in paths]


def _coverage_curve(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    primary_budget_frac: float,
) -> list[dict[str, Any]]:
    durations = metadata["duration"].astype(float).to_numpy()
    similarity, _ = _facility_similarity(embedding)
    rows: list[dict[str, Any]] = []
    budget_fracs = sorted(set([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, float(primary_budget_frac)]))
    rng = np.random.default_rng(seed)
    random_orders = [rng.permutation(len(metadata)).tolist() for _ in range(5)]
    for fraction in budget_fracs:
        count_budget = max(1, int(math.floor(fraction * len(metadata))))
        duration_budget = fraction * float(durations.sum())
        selected, _, _ = select_facility_subset(
            embedding, metadata, count_budget, duration_budget
        )
        rows.append(
            {
                "method": "egotrim",
                "budget_frac": fraction,
                "facility_coverage": facility_value(similarity, selected) / len(metadata),
                "is_primary": math.isclose(fraction, primary_budget_frac),
            }
        )
        random_values = []
        for order in random_orders:
            random_selected = _select_from_order(order, durations, count_budget, duration_budget)
            random_values.append(facility_value(similarity, random_selected) / len(metadata))
        rows.append(
            {
                "method": "uniform_random_mean",
                "budget_frac": fraction,
                "facility_coverage": float(np.mean(random_values)),
                "is_primary": math.isclose(fraction, primary_budget_frac),
            }
        )
    return rows


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "p25", "median", "p75", "max"), map(float, quantiles)))


def _optional_leave_one_out_vendi(
    embedding: np.ndarray, metadata: pd.DataFrame
) -> np.ndarray:
    result = np.full(len(metadata), np.nan)
    for verb, group in metadata.groupby("canonical_verb", sort=True):
        indices = group.index.to_numpy(dtype=int)
        if len(indices) < MIN_VENDI_SAMPLES + 1:
            continue
        kernel, _ = rbf_similarity(embedding[indices])
        full = vendi_from_kernel(kernel)
        for local_index, global_index in enumerate(indices):
            reduced = np.delete(np.delete(kernel, local_index, axis=0), local_index, axis=1)
            result[global_index] = full - vendi_from_kernel(reduced)
    return result


def run_pipeline_from_features(
    valid_metadata: pd.DataFrame,
    trajectory: np.ndarray,
    fingertips: np.ndarray,
    log_duration: np.ndarray,
    rejected_rows: list[dict[str, Any]],
    schema_report: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    if config.enforce_local_isolation:
        output_dir = ensure_output_isolated(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    if len(valid_metadata) < 2:
        raise RuntimeError(
            f"Only {len(valid_metadata)} valid segment(s) remained; at least two are required for PCA. "
            f"Rejected: {len(rejected_rows)}"
        )
    valid_metadata = valid_metadata.reset_index(drop=True).copy()
    model = BalancedEmbeddingModel(config.pca_components, config.pca_variance_target)
    embeddings = model.fit_transform(trajectory, fingertips, log_duration)
    model.save(output_dir / "models")
    np.save(output_dir / "models" / "combined_embeddings.npy", embeddings["combined"])
    write_json(
        output_dir / "models" / "embedding_index.json",
        valid_metadata[["episode_id", "segment_id", "canonical_verb"]].to_dict(orient="records"),
    )

    cluster_ids, cluster_rows, cluster_audit = cluster_within_verbs(
        embeddings["combined"], valid_metadata, config.seed, config.clusterer
    )
    valid_metadata["cluster_id"] = cluster_ids
    score_fields = distinctiveness_scores(
        embeddings["combined"], valid_metadata, cluster_ids, config.neighbor_k
    )
    valid_metadata = pd.concat([valid_metadata, score_fields], axis=1)
    if config.leave_one_out_vendi:
        valid_metadata["diagnostic_nonmonotonic_loo_vendi_delta"] = _optional_leave_one_out_vendi(
            embeddings["combined"], valid_metadata
        )

    neighbors, same_verb_rate = nearest_neighbor_audit(
        embeddings["combined"], valid_metadata, representative_count=20, neighbors_per_query=5
    )
    composition = composition_metrics(valid_metadata["canonical_verb"].astype(str).tolist())
    execution = execution_vendi_by_verb(
        embeddings["combined"], valid_metadata["canonical_verb"].astype(str).tolist()
    )
    ablations = ablation_metrics(embeddings, valid_metadata)
    count_budget = max(1, int(math.floor(config.budget_frac * len(valid_metadata))))
    duration_budget = config.budget_frac * float(valid_metadata["duration"].sum())
    selected, selection_reasons, selection_audit = select_facility_subset(
        embeddings["combined"], valid_metadata, count_budget, duration_budget
    )
    full_similarity, _ = _facility_similarity(embeddings["combined"])
    selected_evaluation = evaluate_subset(
        selected, embeddings["combined"], valid_metadata, full_similarity
    )
    baseline_summary, baseline_runs = run_baselines(
        embeddings["combined"],
        valid_metadata,
        cluster_ids,
        count_budget,
        duration_budget,
        config.seed,
        config.baseline_runs,
    )
    coverage_curve = _coverage_curve(
        embeddings["combined"], valid_metadata, config.seed, config.budget_frac
    )

    valid_metadata["is_valid"] = True
    valid_metadata["rejection_reason"] = ""
    valid_metadata["selected"] = False
    valid_metadata["selection_reason"] = ""
    for index in selected:
        valid_metadata.loc[index, "selected"] = True
        valid_metadata.loc[index, "selection_reason"] = selection_reasons[index]

    valid_score_columns = [
        "episode_id",
        "segment_id",
        "canonical_verb",
        "start_time",
        "end_time",
        "duration",
        "tracking_valid_frac",
        "measured_tracking_valid_frac",
        "tracking_quality",
        "missing_value_rate",
        "interpolated_value_count",
        "coordinate_mode",
        "pose_source_path",
        "video_path",
        "source_data_path",
        "is_valid",
        "rejection_reason",
        "cluster_id",
        "mean_same_verb_knn_distance",
        "distinctiveness_percentile",
        "cluster_rarity",
        "potential_tracking_outlier",
        "selected",
        "selection_reason",
    ]
    if {"start_frame", "end_frame_exclusive"} <= set(valid_metadata.columns):
        duration_position = valid_score_columns.index("duration") + 1
        valid_score_columns[duration_position:duration_position] = [
            "start_frame",
            "end_frame_exclusive",
        ]
    if config.leave_one_out_vendi:
        valid_score_columns.append("diagnostic_nonmonotonic_loo_vendi_delta")
    valid_scores = valid_metadata[valid_score_columns].copy()
    rejected_scores = pd.DataFrame(rejected_rows)
    for column in valid_score_columns:
        if column not in rejected_scores:
            rejected_scores[column] = None
    if len(rejected_scores):
        rejected_scores["is_valid"] = False
        rejected_scores["selected"] = False
    segment_scores = pd.concat(
        [valid_scores, rejected_scores[valid_score_columns]], ignore_index=True
    ).sort_values(["episode_id", "start_time", "segment_id"], kind="stable")
    segment_scores.to_csv(output_dir / "segment_scores.csv", index=False)

    episode_frame = episode_scores(valid_metadata, selected)
    episode_frame.to_csv(output_dir / "episode_scores.csv", index=False)
    cluster_frame = pd.DataFrame(cluster_rows)
    cluster_frame.to_csv(output_dir / "cluster_summary.csv", index=False)
    neighbors.to_csv(output_dir / "nearest_neighbors.csv", index=False)

    manifest_columns = [
        "episode_id",
        "segment_id",
        "canonical_verb",
        "start_time",
        "end_time",
        "duration",
        "cluster_id",
        "distinctiveness_percentile",
        "selection_reason",
        "source_data_path",
    ]
    if {"start_frame", "end_frame_exclusive"} <= set(valid_metadata.columns):
        duration_position = manifest_columns.index("duration") + 1
        manifest_columns[duration_position:duration_position] = [
            "start_frame",
            "end_frame_exclusive",
        ]
    manifest = valid_metadata.iloc[selected][manifest_columns].copy()
    manifest.to_csv(output_dir / "subset_manifest.csv", index=False)
    write_json(output_dir / "subset_manifest.json", manifest.to_dict(orient="records"))

    valid_per_episode = valid_metadata.groupby("episode_id").size()
    fold_per_episode = (
        valid_metadata.assign(_is_fold=valid_metadata["canonical_verb"].eq("fold").astype(int))
        .groupby("episode_id")["_is_fold"]
        .sum()
    )
    sanity = {
        "number_of_episodes_input": int(segment_scores["episode_id"].nunique()),
        "number_of_episodes_with_valid_cycles": int(valid_metadata["episode_id"].nunique()),
        "number_of_valid_cycles": int(len(valid_metadata)),
        "number_of_rejected_cycles": int(len(rejected_rows)),
        "median_cycles_per_episode": float(valid_per_episode.median()),
        "fold_cycles_per_episode": {
            "median": float(fold_per_episode.median()),
            "distribution": {str(k): int(v) for k, v in fold_per_episode.items()},
        },
        "verb_distribution": composition["verb_distribution"],
        "duration_distribution_seconds": _quantiles(valid_metadata["duration"].astype(float)),
        "tracking_valid_distribution": _quantiles(valid_metadata["tracking_quality"].astype(float)),
        "missing_value_rate_distribution": _quantiles(valid_metadata["missing_value_rate"].astype(float)),
        "mean_missing_value_rate": float(valid_metadata["missing_value_rate"].mean()),
        "final_embedding_dimensions": int(embeddings["combined"].shape[1]),
        "trajectory_pca_dimensions": int(embeddings["trajectory"].shape[1]),
        "fingertip_pca_dimensions": int(embeddings["fingertips"].shape[1]),
        "coordinate_modes": valid_metadata["coordinate_mode"].value_counts().to_dict(),
    }
    metrics = {
        "experiment": MODEL_VERSION,
        "seed": config.seed,
        "budget_fraction": config.budget_frac,
        "feature_balance": {
            "trajectory_raw_dimensions": TRAJECTORY_DIMS,
            "fingertip_raw_dimensions": FINGERTIP_DIMS,
            "trajectory_retained_components": model.trajectory_components_,
            "fingertip_retained_components": model.fingertip_components_,
            "duration_dimensions": 1,
            "blocks_standardized_independently": True,
            "final_embedding_standardized": True,
        },
        "composition_diversity": composition,
        "execution_diversity": execution,
        "nearest_neighbors": {
            "representative_query_count": int(neighbors["query_segment_id"].nunique()) if len(neighbors) else 0,
            "same_verb_nearest_neighbor_rate": same_verb_rate,
        },
        "feature_ablations": ablations,
        "clustering": cluster_audit,
        "curation": selection_audit,
        "selected_subset_evaluation": selected_evaluation,
        "baselines": baseline_summary,
        "baseline_runs": baseline_runs,
        "coverage_curve": coverage_curve,
        "data_sanity": sanity,
        "claims_guardrail": (
            "Metrics describe measured composition and execution coverage only; no downstream "
            "robot-policy improvement is claimed."
        ),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "schema_report.json", schema_report)
    chart_paths = generate_charts(
        output_dir / "charts",
        valid_metadata,
        cluster_frame,
        model,
        execution,
        same_verb_rate,
        ablations,
        baseline_summary,
        coverage_curve,
    )
    dashboard = {
        "experiment": MODEL_VERSION,
        "summary": {
            "valid_cycles": len(valid_metadata),
            "rejected_cycles": len(rejected_rows),
            "selected_cycles": len(selected),
            "same_verb_nearest_neighbor_rate": same_verb_rate,
            "composition": composition,
            "weighted_execution_vendi": execution["weighted_mean_execution_vendi"],
        },
        "segments": _jsonable(valid_metadata.to_dict(orient="records")),
        "episodes": _jsonable(episode_frame.to_dict(orient="records")),
        "clusters": _jsonable(cluster_rows),
        "nearest_neighbors": _jsonable(neighbors.to_dict(orient="records")),
        "selected_subset": _jsonable(manifest.to_dict(orient="records")),
        "feature_ablations": ablations,
        "baselines": baseline_summary,
        "coverage_curve": coverage_curve,
        "charts": chart_paths,
    }
    write_json(output_dir / "dashboard_data.json", dashboard)
    return metrics


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    if config.enforce_local_isolation:
        output_dir = ensure_output_isolated(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = output_dir
    try:
        original = load_segments(config.segments)
        schema = discover_segment_schema(original)
        canonical = canonicalize_segments(original, schema)
    except SchemaMappingError as exc:
        report = {
            "status": "failed",
            "stage": "segment_schema_discovery",
            "message": str(exc),
            "details": exc.report,
            "segments_path": str(config.segments),
        }
        write_json(output_dir / "schema_report.json", report)
        raise

    schema_report: dict[str, Any] = {
        "status": "mapped",
        "segments_path": str(config.segments),
        "pose_root": str(config.pose_root),
        "segment_columns_observed": [str(c) for c in original.columns],
        "segment_mapping": asdict(schema),
        "pose_examples": [],
        "repository_convention": (
            "No pre-existing repository code was present. Supported EgoVerse Zarr handling follows "
            "the discovered canonical obs_keypoints/obs_wrist_pose schema and per-frame inverse "
            "obs_head_pose convention."
        ),
    }
    valid_rows: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []
    fingertips: list[np.ndarray] = []
    log_durations: list[float] = []
    rejected: list[dict[str, Any]] = []
    pose_evidence_paths: set[str] = set()
    for row in canonical.to_dict(orient="records"):
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
        if float(row["duration"]) < MIN_DURATION_SECONDS:
            base["rejection_reason"] = f"duration_below_{MIN_DURATION_SECONDS}_seconds"
            rejected.append(base)
            continue
        if float(row["tracking_valid_frac"]) < MIN_TRACKING_VALID_FRAC:
            base["rejection_reason"] = f"declared_tracking_valid_frac_below_{MIN_TRACKING_VALID_FRAC}"
            rejected.append(base)
            continue
        try:
            pose = load_pose_segment(row, config.pose_root, config.coordinate_fallback)
            feature = extract_raw_feature(
                pose,
                float(row["duration"]),
                config.max_interp_gap,
                MIN_TRACKING_VALID_FRAC,
            )
        except TrackingError as exc:
            base["rejection_reason"] = f"tracking_rejected:{exc}"
            rejected.append(base)
            continue
        except SchemaMappingError as exc:
            report = {
                **schema_report,
                "status": "failed",
                "stage": "pose_schema_discovery",
                "message": str(exc),
                "details": exc.report,
                "failed_segment_id": row["segment_id"],
            }
            write_json(output_dir / "schema_report.json", report)
            raise
        valid = dict(row)
        valid.update(
            {
                "measured_tracking_valid_frac": feature.measured_tracking_valid_frac,
                "tracking_quality": min(
                    float(row["tracking_valid_frac"]), feature.measured_tracking_valid_frac
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
        if feature.pose_source_path not in pose_evidence_paths and len(schema_report["pose_examples"]) < 5:
            schema_report["pose_examples"].append(pose.schema_evidence)
            pose_evidence_paths.add(feature.pose_source_path)

    schema_report["valid_segment_count"] = len(valid_rows)
    schema_report["rejected_segment_count"] = len(rejected)
    return run_pipeline_from_features(
        pd.DataFrame(valid_rows),
        np.asarray(trajectories, dtype=float),
        np.asarray(fingertips, dtype=float),
        np.asarray(log_durations, dtype=float),
        rejected,
        schema_report,
        config,
    )


def make_synthetic_dataset(root: Path, seed: int = 42, episodes: int = 8) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    pose_root = root / "poses"
    pose_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    verbs = ("fold", "smooth", "pick", "adjust")
    rows: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        episode_id = f"synthetic_ep_{episode_index:02d}"
        for verb_index, verb in enumerate(verbs):
            segment_id = f"{episode_id}_{verb}_{verb_index}"
            duration = 0.9 + 0.12 * verb_index + 0.03 * (episode_index % 3)
            frame_count = 42 + episode_index % 5
            time = np.linspace(0.0, duration, frame_count)
            phase = (episode_index % 2) * 0.35
            wrists = np.zeros((frame_count, 2, 3), dtype=float)
            for hand in range(2):
                side = -1.0 if hand == 0 else 1.0
                if verb == "fold":
                    wrists[:, hand, 0] = 0.35 + side * (0.24 - 0.16 * time / duration)
                    wrists[:, hand, 1] = 0.25 + 0.04 * np.sin(np.pi * time / duration + phase)
                    wrists[:, hand, 2] = 0.55 + 0.05 * np.sin(np.pi * time / duration)
                elif verb == "smooth":
                    wrists[:, hand, 0] = 0.35 + side * 0.18 + 0.14 * np.sin(2 * np.pi * time / duration + phase)
                    wrists[:, hand, 1] = 0.28 + 0.03 * np.cos(2 * np.pi * time / duration)
                    wrists[:, hand, 2] = 0.50 + 0.01 * hand
                elif verb == "pick":
                    wrists[:, hand, 0] = 0.35 + side * 0.15
                    wrists[:, hand, 1] = 0.24 + 0.02 * np.sin(np.pi * time / duration)
                    wrists[:, hand, 2] = 0.48 + 0.20 * time / duration
                else:
                    wrists[:, hand, 0] = 0.35 + side * 0.15 + 0.05 * np.cos(3 * np.pi * time / duration)
                    wrists[:, hand, 1] = 0.25 + 0.08 * time / duration
                    wrists[:, hand, 2] = 0.52 + 0.04 * np.sin(2 * np.pi * time / duration + hand)
            wrists += rng.normal(0.0, 0.0025, size=wrists.shape)
            joints = np.repeat(wrists[:, :, None, :], 21, axis=2)
            for joint in range(21):
                finger = min(4, max(0, (joint - 1) // 4))
                depth = 0.008 * (joint % 4 + 1)
                joints[:, :, joint, 0] += (finger - 2) * 0.012
                joints[:, :, joint, 1] += depth
                joints[:, :, joint, 2] += 0.01 * np.sin(
                    (verb_index + 1) * np.pi * time / duration + finger * 0.2
                )[:, None]
            if episode_index == 0 and verb == "adjust":
                joints[20, 0, 8, :] = np.nan  # safe one-frame internal gap
            pose_name = f"{segment_id}.npz"
            np.savez_compressed(
                pose_root / pose_name,
                hand_joints=joints,
                timestamps=time,
                coordinate_frame=np.asarray("head"),
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "canonical_verb": verb,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration,
                    "tracking_valid_frac": 0.98,
                    "pose_ref": pose_name,
                    "video_path": "",
                    "source_data_path": str(pose_root / pose_name),
                }
            )
    segments_path = root / "segments.csv"
    pd.DataFrame(rows).to_csv(segments_path, index=False)
    return segments_path, pose_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, help="Segment manifest (CSV/Parquet/JSON/JSONL)")
    parser.add_argument("--pose-root", type=Path, help="Root containing referenced pose stores")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-frac", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-interp-gap", type=int, default=2)
    parser.add_argument("--pca-components", type=int, default=15)
    parser.add_argument("--pca-variance-target", type=float, default=0.99)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--baseline-runs", type=int, default=10)
    parser.add_argument("--clusterer", choices=("kmeans", "hdbscan"), default="kmeans")
    parser.add_argument(
        "--coordinate-fallback",
        choices=("error", "already_head_frame"),
        default="error",
        help="Explicit override only when no head transform/metadata exists",
    )
    parser.add_argument("--leave-one-out-vendi", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Generate isolated synthetic poses and run the entire pipeline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.budget_frac <= 1:
        raise SystemExit("--budget-frac must be in (0, 1]")
    if args.max_interp_gap < 0 or args.pca_components < 1 or args.baseline_runs < 1:
        raise SystemExit("gap/components/baseline-runs must be positive (gap may be zero)")
    output_dir = ensure_output_isolated(args.output_dir)
    if args.synthetic_smoke:
        segments, pose_root = make_synthetic_dataset(
            output_dir / "synthetic_inputs", args.seed
        )
    else:
        if args.segments is None or args.pose_root is None:
            raise SystemExit("--segments and --pose-root are required unless --synthetic-smoke is used")
        segments, pose_root = args.segments.resolve(), args.pose_root.resolve()
    config = PipelineConfig(
        segments=segments,
        pose_root=pose_root,
        output_dir=output_dir,
        budget_frac=args.budget_frac,
        seed=args.seed,
        max_interp_gap=args.max_interp_gap,
        pca_components=args.pca_components,
        pca_variance_target=args.pca_variance_target,
        neighbor_k=args.neighbor_k,
        baseline_runs=args.baseline_runs,
        clusterer=args.clusterer,
        coordinate_fallback=args.coordinate_fallback,
        leave_one_out_vendi=args.leave_one_out_vendi,
    )
    try:
        metrics = run_pipeline(config)
    except SchemaMappingError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        print(f"See {output_dir / 'schema_report.json'}", file=sys.stderr)
        return 2
    except (TrackingError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"PIPELINE ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "valid_cycles": metrics["data_sanity"]["number_of_valid_cycles"],
                "rejected_cycles": metrics["data_sanity"]["number_of_rejected_cycles"],
                "selected_cycles": metrics["curation"]["selected_count"],
                "same_verb_nearest_neighbor_rate": metrics["nearest_neighbors"][
                    "same_verb_nearest_neighbor_rate"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
