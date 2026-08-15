"""Extract physical attempt features and curate redundant EgoVerse attempts.

This stage consumes attempt boundaries; it does not create or alter them.

Smoke test one episode and then compare all available extracted features:
    modal run modal_attempt_features.py --max-episodes 1

Process every episode that has attempts.json:
    modal run modal_attempt_features.py --max-episodes 0

Outputs:
    /egoverse/attempt_features/<episode-id>/features.json
    /egoverse/attempt_similarity/summary.json
    /egoverse/attempt_similarity/tasks/<task-key>.json
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
EPISODES = MOUNT / "episodes"
ATTEMPTS = MOUNT / "attempts"
FEATURES = MOUNT / "attempt_features"
SIMILARITY = MOUNT / "attempt_similarity"

app = modal.App("egoverse-attempt-features")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.2.6",
    "scipy==1.15.3",
    "zarr==3.1.5",
)


@dataclass(frozen=True)
class SimilarityConfig:
    trajectory_weight: float = 0.45
    orientation_weight: float = 0.25
    coordination_weight: float = 0.20
    dynamics_weight: float = 0.10
    trajectory_scale_m: float = 0.15
    orientation_scale_rad: float = math.pi / 4.0
    coordination_scale_m: float = 0.15
    dynamics_scale: float = 0.50
    redundancy_threshold: float = 0.90

    def validate(self) -> None:
        weights = (
            self.trajectory_weight,
            self.orientation_weight,
            self.coordination_weight,
            self.dynamics_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Similarity weights must be non-negative and sum above zero")
        scales = (
            self.trajectory_scale_m,
            self.orientation_scale_rad,
            self.coordination_scale_m,
            self.dynamics_scale,
        )
        if any(scale <= 0 for scale in scales):
            raise ValueError("Similarity scales must be positive")
        if not 0 <= self.redundancy_threshold <= 1:
            raise ValueError("redundancy_threshold must be in [0, 1]")


def _timestamps_seconds(store, frame_count: int, fps: float):
    import numpy as np

    nominal_delta = 1.0 / fps
    for key, scale in (
        ("obs_rgb_timestamps_ns", 1e-9),
        ("timestamps_ns", 1e-9),
        ("timestamps", 1.0),
    ):
        if key not in store:
            continue
        values = np.asarray(store[key][:frame_count], dtype=np.float64).reshape(-1)
        if len(values) != frame_count or not np.all(np.isfinite(values)):
            continue
        seconds = (values - values[0]) * scale
        if np.all(np.diff(seconds) > 0):
            deltas = np.diff(seconds)
            plausible = (deltas >= nominal_delta * 0.25) & (deltas <= nominal_delta * 4.0)
            replacement = np.median(deltas[plausible]) if np.any(plausible) else nominal_delta
            sanitized = np.concatenate(([0.0], np.cumsum(np.where(plausible, deltas, replacement))))
            corrected = int(np.count_nonzero(~plausible))
            source = key if corrected == 0 else f"{key}_sanitized_{corrected}_deltas"
            return sanitized, source
    return np.arange(frame_count, dtype=np.float64) / fps, "fps_fallback"


def _stored_quat_to_rotation(quaternions):
    """Convert EgoVerse qw,qx,qy,qz quaternions to SciPy rotations."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("Pose stream contains invalid quaternions")
    unit = quaternions / norms
    return Rotation.from_quat(unit[:, [1, 2, 3, 0]])


def _head_relative_hands(left_pose, right_pose, head_pose):
    import numpy as np

    head_rotation = _stored_quat_to_rotation(head_pose[:, 3:7])
    inverse_head = head_rotation.inv()
    left_rotation = _stored_quat_to_rotation(left_pose[:, 3:7])
    right_rotation = _stored_quat_to_rotation(right_pose[:, 3:7])
    left_xyz = inverse_head.apply(np.asarray(left_pose[:, :3]) - np.asarray(head_pose[:, :3]))
    right_xyz = inverse_head.apply(np.asarray(right_pose[:, :3]) - np.asarray(head_pose[:, :3]))
    return left_xyz, right_xyz, inverse_head * left_rotation, inverse_head * right_rotation


def _resample_positions(values, timestamps, normalized_samples: int):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(values) == 1:
        return np.repeat(values, normalized_samples, axis=0)
    source = (timestamps - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1e-12)
    target = np.linspace(0.0, 1.0, normalized_samples)
    return np.column_stack([np.interp(target, source, values[:, column]) for column in range(values.shape[1])])


def _resample_rotations(rotations, timestamps, normalized_samples: int):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(rotations) == 1:
        return Rotation.concatenate([rotations] * normalized_samples)
    source = (timestamps - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1e-12)
    target = np.linspace(0.0, 1.0, normalized_samples)
    return Slerp(source, rotations)(target)


def _rotation_to_6d(rotations):
    """Store the first two rotation-matrix columns (continuous 6D representation)."""
    matrices = rotations.as_matrix()
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(len(matrices), 6)


def _speeds(positions, timestamps):
    import numpy as np

    if len(positions) < 2:
        return np.zeros(1, dtype=np.float64)
    delta_t = np.maximum(np.diff(timestamps), 1e-8)
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) / delta_t


def _angular_speeds(rotations, timestamps):
    import numpy as np

    if len(rotations) < 2:
        return np.zeros(1, dtype=np.float64)
    delta_t = np.maximum(np.diff(timestamps), 1e-8)
    angles = np.linalg.norm((rotations[:-1].inv() * rotations[1:]).as_rotvec(), axis=1)
    return angles / delta_t


def _runs(mask):
    start = 0
    for index in range(1, len(mask) + 1):
        if index == len(mask) or bool(mask[index]) != bool(mask[start]):
            yield start, index, bool(mask[start])
            start = index


def _pause_statistics(stationary, timestamps, minimum_pause_seconds: float):
    import numpy as np

    if len(stationary) == 0:
        return 0, 0.0
    interval_durations = np.diff(timestamps)
    durations = []
    for start, end, is_stationary in _runs(stationary):
        if not is_stationary:
            continue
        duration = float(interval_durations[start:end].sum())
        if duration >= minimum_pause_seconds:
            durations.append(duration)
    return len(durations), float(np.mean(durations)) if durations else 0.0


def _hand_dynamics(positions, rotations, timestamps):
    import numpy as np

    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angular_displacement = np.linalg.norm((rotations[:-1].inv() * rotations[1:]).as_rotvec(), axis=1)
    speed = _speeds(positions, timestamps)
    angular_speed = _angular_speeds(rotations, timestamps)
    return {
        "path_length": float(displacement.sum()),
        "mean_speed": float(speed.mean()),
        "max_speed": float(speed.max()),
        "speed_variance": float(speed.var()),
        "mean_angular_speed": float(angular_speed.mean()),
        "total_angular_rotation": float(angular_displacement.sum()),
    }


def _validate_attempt_bounds(attempt: dict, frame_count: int) -> tuple[int, int]:
    start = int(attempt["start_idx"])
    end = int(attempt["end_idx"])
    if start < 0 or end > frame_count or end - start < 2:
        raise ValueError(f"Invalid attempt range [{start}, {end}) for {frame_count} frames")
    return start, end


def build_attempt_features(
    *,
    episode_id: str,
    task_id: str,
    attempt: dict,
    left_pose,
    right_pose,
    head_pose,
    timestamps,
    normalized_samples: int = 32,
    activity_velocity_threshold: float = 0.05,
    minimum_pause_seconds: float = 0.30,
) -> dict:
    """Build a JSON-serializable physical representation for one attempt."""
    import numpy as np

    if normalized_samples < 2:
        raise ValueError("normalized_samples must be at least 2")
    left_xyz, right_xyz, left_rotation, right_rotation = _head_relative_hands(
        left_pose, right_pose, head_pose
    )
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        raise ValueError("Attempt timestamps must span a positive duration")

    left_speed = _speeds(left_xyz, timestamps)
    right_speed = _speeds(right_xyz, timestamps)
    left_active = left_speed >= activity_velocity_threshold
    right_active = right_speed >= activity_velocity_threshold
    stationary = ~(left_active | right_active)
    pause_count, mean_pause_duration = _pause_statistics(
        stationary, timestamps, minimum_pause_seconds
    )

    left_normalized = _resample_positions(left_xyz, timestamps, normalized_samples)
    right_normalized = _resample_positions(right_xyz, timestamps, normalized_samples)
    left_rotation_normalized = _resample_rotations(left_rotation, timestamps, normalized_samples)
    right_rotation_normalized = _resample_rotations(right_rotation, timestamps, normalized_samples)
    inter_hand_distance = np.linalg.norm(right_normalized - left_normalized, axis=1)
    relative_hand_position = right_normalized - left_normalized

    left_stats = _hand_dynamics(left_xyz, left_rotation, timestamps)
    right_stats = _hand_dynamics(right_xyz, right_rotation, timestamps)
    left_path = left_stats["path_length"]
    right_path = right_stats["path_length"]
    handedness = (right_path - left_path) / (right_path + left_path + 1e-8)
    attempt_number = int(attempt["attempt_id"])

    return {
        "attempt_id": f"{episode_id}:{attempt_number}",
        "episode_id": episode_id,
        "episode_attempt_id": attempt_number,
        "task_id": task_id,
        "source_range": {
            "start_idx": int(attempt["start_idx"]),
            "end_idx": int(attempt["end_idx"]),
            "start_sec": float(attempt.get("start_sec", timestamps[0])),
            "end_sec": float(attempt.get("end_sec", timestamps[-1])),
        },
        "normalization": {
            "coordinate_frame": "head_relative",
            "normalized_samples": normalized_samples,
            "orientation_storage": "rotation_6d_first_two_columns",
        },
        "trajectory": {
            "left_xyz": left_normalized.tolist(),
            "right_xyz": right_normalized.tolist(),
        },
        "orientation": {
            "left": _rotation_to_6d(left_rotation_normalized).tolist(),
            "right": _rotation_to_6d(right_rotation_normalized).tolist(),
        },
        "coordination": {
            "inter_hand_distance": inter_hand_distance.tolist(),
            "relative_hand_position": relative_hand_position.tolist(),
        },
        "activity": {
            "velocity_threshold_mps": activity_velocity_threshold,
            "handedness": float(handedness),
            "left_active_fraction": float(left_active.mean()),
            "right_active_fraction": float(right_active.mean()),
            "both_active_fraction": float((left_active & right_active).mean()),
        },
        "dynamics": {
            "duration": duration,
            "left": left_stats,
            "right": right_stats,
            "stationary_fraction": float(stationary.mean()),
            "meaningful_pause_count": pause_count,
            "mean_pause_duration": mean_pause_duration,
            "minimum_pause_seconds": minimum_pause_seconds,
        },
    }


def _rotation_6d_to_matrices(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64).reshape(-1, 2, 3)
    first = values[:, 0]
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = values[:, 1] - np.sum(first * values[:, 1], axis=1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=2)


def _distance_similarity(distance: float, scale: float) -> float:
    return float(math.exp(-max(0.0, distance) / scale))


def _trajectory_similarity(first: dict, second: dict, scale: float) -> tuple[float, float]:
    import numpy as np

    left_distance = np.linalg.norm(
        np.asarray(first["trajectory"]["left_xyz"]) - np.asarray(second["trajectory"]["left_xyz"]),
        axis=1,
    )
    right_distance = np.linalg.norm(
        np.asarray(first["trajectory"]["right_xyz"]) - np.asarray(second["trajectory"]["right_xyz"]),
        axis=1,
    )
    distance = float(np.mean((left_distance + right_distance) / 2.0))
    return _distance_similarity(distance, scale), distance


def _orientation_similarity(first: dict, second: dict, scale: float) -> tuple[float, float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    distances = []
    for hand in ("left", "right"):
        first_rotation = Rotation.from_matrix(_rotation_6d_to_matrices(first["orientation"][hand]))
        second_rotation = Rotation.from_matrix(_rotation_6d_to_matrices(second["orientation"][hand]))
        distances.append(np.linalg.norm((first_rotation.inv() * second_rotation).as_rotvec(), axis=1))
    distance = float(np.mean(np.concatenate(distances)))
    return _distance_similarity(distance, scale), distance


def _coordination_similarity(first: dict, second: dict, scale: float) -> tuple[float, float]:
    import numpy as np

    first_coordination = first["coordination"]
    second_coordination = second["coordination"]
    distance_difference = np.abs(
        np.asarray(first_coordination["inter_hand_distance"])
        - np.asarray(second_coordination["inter_hand_distance"])
    )
    position_difference = np.linalg.norm(
        np.asarray(first_coordination["relative_hand_position"])
        - np.asarray(second_coordination["relative_hand_position"]),
        axis=1,
    )
    distance = float(np.mean((distance_difference + position_difference) / 2.0))
    return _distance_similarity(distance, scale), distance


def _symmetric_log_difference(first: float, second: float, floor: float = 1e-6) -> float:
    return abs(math.log(max(first, floor) / max(second, floor)))


def _dynamics_distance(first: dict, second: dict) -> float:
    differences = []
    first_dynamics, second_dynamics = first["dynamics"], second["dynamics"]
    # Duration is deliberately one low-weight member of a larger feature set.
    differences.append(0.5 * _symmetric_log_difference(first_dynamics["duration"], second_dynamics["duration"]))
    for hand in ("left", "right"):
        for key in (
            "path_length",
            "mean_speed",
            "max_speed",
            "speed_variance",
            "mean_angular_speed",
            "total_angular_rotation",
        ):
            differences.append(
                _symmetric_log_difference(first_dynamics[hand][key], second_dynamics[hand][key])
            )
    first_activity, second_activity = first["activity"], second["activity"]
    for key in (
        "handedness",
        "left_active_fraction",
        "right_active_fraction",
        "both_active_fraction",
    ):
        differences.append(abs(float(first_activity[key]) - float(second_activity[key])))
    differences.append(abs(first_dynamics["stationary_fraction"] - second_dynamics["stationary_fraction"]))
    return sum(differences) / len(differences)


def compare_attempts(first: dict, second: dict, config: SimilarityConfig) -> dict:
    """Return interpretable component and overall similarities in [0, 1]."""
    if first["task_id"] != second["task_id"]:
        raise ValueError("Attempts can only be compared within the same task")
    config.validate()
    trajectory, trajectory_distance = _trajectory_similarity(
        first, second, config.trajectory_scale_m
    )
    orientation, orientation_distance = _orientation_similarity(
        first, second, config.orientation_scale_rad
    )
    coordination, coordination_distance = _coordination_similarity(
        first, second, config.coordination_scale_m
    )
    dynamics_distance = _dynamics_distance(first, second)
    dynamics = _distance_similarity(dynamics_distance, config.dynamics_scale)
    weighted = (
        config.trajectory_weight * trajectory
        + config.orientation_weight * orientation
        + config.coordination_weight * coordination
        + config.dynamics_weight * dynamics
    )
    weight_sum = (
        config.trajectory_weight
        + config.orientation_weight
        + config.coordination_weight
        + config.dynamics_weight
    )
    return {
        "attempt_a": first["attempt_id"],
        "attempt_b": second["attempt_id"],
        "trajectory_similarity": trajectory,
        "orientation_similarity": orientation,
        "coordination_similarity": coordination,
        "dynamics_similarity": dynamics,
        "overall_similarity": float(weighted / weight_sum),
        "distances": {
            "trajectory_m": trajectory_distance,
            "orientation_rad": orientation_distance,
            "coordination_m": coordination_distance,
            "dynamics_normalized": dynamics_distance,
        },
    }


def greedy_curate(attempts: list[dict], pairwise: list[dict], threshold: float) -> list[dict]:
    if not attempts:
        return []
    similarities = {
        frozenset((pair["attempt_a"], pair["attempt_b"])): pair["overall_similarity"]
        for pair in pairwise
    }
    kept = []
    decisions = []
    for attempt in attempts:
        attempt_id = attempt["attempt_id"]
        if not kept:
            kept.append(attempt_id)
            decisions.append(
                {"attempt_id": attempt_id, "decision": "KEEP", "represented_by": attempt_id, "similarity": 1.0}
            )
            continue
        representative, score = max(
            ((candidate, similarities[frozenset((attempt_id, candidate))]) for candidate in kept),
            key=lambda item: item[1],
        )
        decision = "DROP" if score >= threshold else "KEEP"
        if decision == "KEEP":
            kept.append(attempt_id)
            representative, score = attempt_id, 1.0
        decisions.append(
            {
                "attempt_id": attempt_id,
                "decision": decision,
                "represented_by": representative,
                "similarity": float(score),
            }
        )
    return decisions


def coverage_metrics(decisions: list[dict]) -> dict:
    import numpy as np

    if not decisions:
        return {
            "attempts_originally": 0,
            "attempts_kept": 0,
            "retained_fraction": 0.0,
            "reduction_fraction": 0.0,
            "mean_behavioral_coverage": 0.0,
            "worst_case_coverage": 0.0,
        }
    coverage = np.asarray([decision["similarity"] for decision in decisions], dtype=np.float64)
    kept = sum(decision["decision"] == "KEEP" for decision in decisions)
    return {
        "attempts_originally": len(decisions),
        "attempts_kept": kept,
        "retained_fraction": kept / len(decisions),
        "reduction_fraction": 1.0 - kept / len(decisions),
        "mean_behavioral_coverage": float(coverage.mean()),
        "worst_case_coverage": float(coverage.min()),
    }


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)


def _task_key(task_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-")[:60] or "task"
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:10]
    return f"{slug}-{digest}"


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    cpu=4,
    memory=8192,
    max_containers=50,
    timeout=60 * 30,
)
def extract_episode_features(
    episode_id: str,
    normalized_samples: int = 32,
    activity_velocity_threshold: float = 0.05,
    minimum_pause_seconds: float = 0.30,
) -> dict:
    import numpy as np
    import zarr

    volume.reload()
    attempt_path = ATTEMPTS / episode_id / "attempts.json"
    if not attempt_path.is_file():
        raise FileNotFoundError(f"Attempt manifest not found: {attempt_path}")
    attempt_manifest = json.loads(attempt_path.read_text())
    store = zarr.open_group(str(EPISODES / episode_id), mode="r")
    required = ("left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose")
    missing = [key for key in required if key not in store]
    if missing:
        raise KeyError(f"{episode_id} is missing required pose arrays: {missing}")

    fps = float(store.attrs.get("fps", 30.0))
    frame_count = min(
        int(store.attrs.get("total_frames", store[required[0]].shape[0])),
        *(int(store[key].shape[0]) for key in required),
    )
    timestamps, timestamp_source = _timestamps_seconds(store, frame_count, fps)
    task_id = str(attempt_manifest.get("task") or store.attrs.get("task_description") or store.attrs.get("task_name") or "unknown task")
    features = []
    errors = []
    for attempt in attempt_manifest.get("attempts", []):
        try:
            start, end = _validate_attempt_bounds(attempt, frame_count)
            features.append(
                build_attempt_features(
                    episode_id=episode_id,
                    task_id=task_id,
                    attempt=attempt,
                    left_pose=np.asarray(store[required[0]][start:end], dtype=np.float64),
                    right_pose=np.asarray(store[required[1]][start:end], dtype=np.float64),
                    head_pose=np.asarray(store[required[2]][start:end], dtype=np.float64),
                    timestamps=timestamps[start:end],
                    normalized_samples=normalized_samples,
                    activity_velocity_threshold=activity_velocity_threshold,
                    minimum_pause_seconds=minimum_pause_seconds,
                )
            )
        except (ValueError, KeyError) as error:
            errors.append({"attempt_id": attempt.get("attempt_id"), "error": str(error)})

    payload = {
        "schema_version": 1,
        "episode_id": episode_id,
        "task_id": task_id,
        "timestamp_source": timestamp_source,
        "parameters": {
            "normalized_samples": normalized_samples,
            "activity_velocity_threshold": activity_velocity_threshold,
            "minimum_pause_seconds": minimum_pause_seconds,
        },
        "attempt_count": len(features),
        "errors": errors,
        "attempts": features,
    }
    _write_json_atomic(FEATURES / episode_id / "features.json", payload)
    volume.commit()
    result = {
        "episode_id": episode_id,
        "task_id": task_id,
        "attempt_count": len(features),
        "error_count": len(errors),
    }
    print(json.dumps(result))
    return result


@app.function(image=image, volumes={str(MOUNT): volume}, cpu=4, memory=8192, timeout=60 * 60)
def compare_all_features(
    config_payload: dict,
    coverage_thresholds: list[float],
) -> dict:
    volume.reload()
    config = SimilarityConfig(**config_payload)
    config.validate()
    by_task: dict[str, list[dict]] = {}
    for feature_path in sorted(FEATURES.glob("*/features.json")):
        payload = json.loads(feature_path.read_text())
        for attempt in payload.get("attempts", []):
            by_task.setdefault(attempt["task_id"], []).append(attempt)

    task_summaries = []
    all_decisions = []
    curve_decisions = {threshold: [] for threshold in coverage_thresholds}
    for task_id, attempts in sorted(by_task.items()):
        attempts.sort(key=lambda value: (value["episode_id"], value["episode_attempt_id"]))
        pairwise = [
            compare_attempts(attempts[first], attempts[second], config)
            for first in range(len(attempts))
            for second in range(first + 1, len(attempts))
        ]
        decisions = greedy_curate(attempts, pairwise, config.redundancy_threshold)
        all_decisions.extend(decisions)
        threshold_curve = []
        for threshold in coverage_thresholds:
            threshold_decisions = greedy_curate(attempts, pairwise, threshold)
            curve_decisions[threshold].extend(threshold_decisions)
            threshold_curve.append({"threshold": threshold, **coverage_metrics(threshold_decisions)})
        task_payload = {
            "schema_version": 1,
            "task_id": task_id,
            "similarity_config": asdict(config),
            "attempt_ids": [attempt["attempt_id"] for attempt in attempts],
            "pairwise": pairwise,
            "curation": decisions,
            "coverage": coverage_metrics(decisions),
            "threshold_curve": threshold_curve,
        }
        task_path = SIMILARITY / "tasks" / f"{_task_key(task_id)}.json"
        _write_json_atomic(task_path, task_payload)
        task_summaries.append(
            {"task_id": task_id, "result_path": str(task_path), **coverage_metrics(decisions)}
        )

    dataset_coverage = coverage_metrics(all_decisions)
    summary = {
        "schema_version": 1,
        "similarity_config": asdict(config),
        "task_count": len(task_summaries),
        **dataset_coverage,
        "threshold_curve": [
            {"threshold": threshold, **coverage_metrics(curve_decisions[threshold])}
            for threshold in coverage_thresholds
        ],
        "tasks": task_summaries,
    }
    _write_json_atomic(SIMILARITY / "summary.json", summary)
    volume.commit()
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}))
    return summary


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 5)
def list_attempt_episode_ids(max_episodes: int) -> list[str]:
    volume.reload()
    episode_ids = sorted(
        path.parent.name for path in ATTEMPTS.glob("*/attempts.json") if path.is_file()
    )
    return episode_ids if max_episodes <= 0 else episode_ids[:max_episodes]


@app.local_entrypoint()
def main(
    max_episodes: int = 1,
    normalized_samples: int = 32,
    activity_velocity_threshold: float = 0.05,
    minimum_pause_seconds: float = 0.30,
    redundancy_threshold: float = 0.90,
    trajectory_weight: float = 0.45,
    orientation_weight: float = 0.25,
    coordination_weight: float = 0.20,
    dynamics_weight: float = 0.10,
    trajectory_scale_m: float = 0.15,
    orientation_scale_rad: float = math.pi / 4.0,
    coordination_scale_m: float = 0.15,
    dynamics_scale: float = 0.50,
    coverage_thresholds: str = "0.70,0.75,0.80,0.85,0.90,0.95",
):
    episode_ids = list_attempt_episode_ids.remote(max_episodes)
    if not episode_ids:
        raise RuntimeError("No /egoverse/attempts/*/attempts.json manifests were found")
    print(f"Extracting physical features for {len(episode_ids)} episode(s)")
    for result in extract_episode_features.map(
        episode_ids,
        kwargs={
            "normalized_samples": normalized_samples,
            "activity_velocity_threshold": activity_velocity_threshold,
            "minimum_pause_seconds": minimum_pause_seconds,
        },
    ):
        print(result)

    config = SimilarityConfig(
        trajectory_weight=trajectory_weight,
        orientation_weight=orientation_weight,
        coordination_weight=coordination_weight,
        dynamics_weight=dynamics_weight,
        trajectory_scale_m=trajectory_scale_m,
        orientation_scale_rad=orientation_scale_rad,
        coordination_scale_m=coordination_scale_m,
        dynamics_scale=dynamics_scale,
        redundancy_threshold=redundancy_threshold,
    )
    thresholds = [float(value.strip()) for value in coverage_thresholds.split(",") if value.strip()]
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("All coverage thresholds must be in [0, 1]")
    summary = compare_all_features.remote(asdict(config), thresholds)
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=2))
