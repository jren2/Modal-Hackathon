"""Create kinematic and fixed-baseline segment manifests for EgoVerse episodes.

The source Zarrs are read-only. Results are written to:
    /egoverse/kinematic_segments/<episode-id>/kinematic.json
    /egoverse/kinematic_segments/<episode-id>/fixed_1s.json

Run on the first two episodes:
    modal run modal_kinematic_segment.py --max-episodes 2
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
EPISODES = MOUNT / "episodes"
OUTPUT = MOUNT / "kinematic_segments"

app = modal.App("egoverse-kinematic-segmentation")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.2.6",
    "scipy==1.15.3",
    "ruptures==1.1.10",
    "zarr==3.1.5",
)


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
            # Aria streams can contain acquisition-clock discontinuities that
            # are not present in constant-FPS video playback. Preserve ordinary
            # jitter but replace implausible deltas before computing velocities.
            deltas = np.diff(seconds)
            plausible = (deltas >= nominal_delta * 0.25) & (deltas <= nominal_delta * 4.0)
            replacement = np.median(deltas[plausible]) if np.any(plausible) else nominal_delta
            sanitized = np.concatenate(([0.0], np.cumsum(np.where(plausible, deltas, replacement))))
            corrected = int(np.count_nonzero(~plausible))
            source = key if corrected == 0 else f"{key}_sanitized_{corrected}_deltas"
            return sanitized, source
    return np.arange(frame_count, dtype=np.float64) / fps, "fps_fallback"


def _stored_quat_to_rotation(quaternions):
    """EgoVerse stores qw,qx,qy,qz; SciPy expects qx,qy,qz,qw."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("Pose stream contains invalid quaternions")
    unit = quaternions / norms
    return Rotation.from_quat(unit[:, [1, 2, 3, 0]])


def _head_relative_pose(hand_pose, head_pose):
    import numpy as np

    hand_position = np.asarray(hand_pose[:, :3], dtype=np.float64)
    head_position = np.asarray(head_pose[:, :3], dtype=np.float64)
    hand_rotation = _stored_quat_to_rotation(hand_pose[:, 3:7])
    head_rotation = _stored_quat_to_rotation(head_pose[:, 3:7])
    inverse_head = head_rotation.inv()
    relative_position = inverse_head.apply(hand_position - head_position)
    relative_rotation = inverse_head * hand_rotation
    return relative_position, relative_rotation


def _differentiate_positions(positions, timestamps):
    import numpy as np

    delta_t = np.diff(timestamps)
    fallback_dt = np.median(delta_t[delta_t > 0])
    delta_t = np.maximum(delta_t, fallback_dt * 0.1)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / delta_t
    return np.concatenate(([speed[0]], speed))


def _angular_speed(rotations, timestamps):
    import numpy as np

    delta_t = np.diff(timestamps)
    fallback_dt = np.median(delta_t[delta_t > 0])
    delta_t = np.maximum(delta_t, fallback_dt * 0.1)
    relative = rotations[:-1].inv() * rotations[1:]
    speed = np.linalg.norm(relative.as_rotvec(), axis=1) / delta_t
    return np.concatenate(([speed[0]], speed))


def _smooth(features, window: int):
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    if window <= 1:
        return features
    return uniform_filter1d(features, size=window, axis=0, mode="nearest")


def _robust_scale(features):
    """Median/MAD scaling prevents spikes from setting every feature's scale."""
    import numpy as np

    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0) * 1.4826
    standard_deviation = np.std(features, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(standard_deviation > 1e-8, standard_deviation, 1.0))
    return (features - median) / scale, median, scale


def _kinematic_features(store, frame_count: int, fps: float, smoothing_frames: int, head_weight: float):
    import numpy as np

    required = ("left.obs_ee_pose", "right.obs_ee_pose")
    missing = [key for key in required if key not in store]
    if missing:
        raise KeyError(f"Missing required pose arrays: {missing}")

    left = np.asarray(store[required[0]][:frame_count], dtype=np.float64)
    right = np.asarray(store[required[1]][:frame_count], dtype=np.float64)
    if "obs_head_pose" in store:
        head = np.asarray(store["obs_head_pose"][:frame_count], dtype=np.float64)
        left_position, left_rotation = _head_relative_pose(left, head)
        right_position, right_rotation = _head_relative_pose(right, head)
        normalized_to_head = True
    else:
        # Some embodiments do not provide a head pose. Keep the episode usable,
        # but record the fallback prominently in the manifest.
        left_position, right_position = left[:, :3], right[:, :3]
        left_rotation = _stored_quat_to_rotation(left[:, 3:7])
        right_rotation = _stored_quat_to_rotation(right[:, 3:7])
        head = None
        normalized_to_head = False

    timestamps, timestamp_source = _timestamps_seconds(store, frame_count, fps)
    columns = [
        _differentiate_positions(left_position, timestamps),
        _differentiate_positions(right_position, timestamps),
        _angular_speed(left_rotation, timestamps),
        _angular_speed(right_rotation, timestamps),
        np.linalg.norm(left_position - right_position, axis=1),
    ]
    feature_names = [
        "left_linear_speed",
        "right_linear_speed",
        "left_angular_speed",
        "right_angular_speed",
        "inter_hand_distance",
    ]
    if head is not None and head_weight > 0:
        head_rotation = _stored_quat_to_rotation(head[:, 3:7])
        columns.extend(
            [
                head_weight * _differentiate_positions(head[:, :3], timestamps),
                head_weight * _angular_speed(head_rotation, timestamps),
            ]
        )
        feature_names.extend(["weighted_head_linear_speed", "weighted_head_angular_speed"])

    smoothed = _smooth(np.column_stack(columns), smoothing_frames)
    scaled, centers, scales = _robust_scale(smoothed)
    return scaled, timestamps, {
        "feature_names": feature_names,
        "normalization_center": centers.tolist(),
        "normalization_scale": scales.tolist(),
        "head_relative": normalized_to_head,
        "timestamp_source": timestamp_source,
    }


def _merge_short_segments(boundaries: list[int], features, minimum_frames: int) -> list[int]:
    import numpy as np

    boundaries = list(boundaries)
    while len(boundaries) > 2:
        lengths = np.diff(boundaries)
        short_index = int(np.argmin(lengths))
        if lengths[short_index] >= minimum_frames:
            break
        if short_index == 0:
            del boundaries[1]
        elif short_index == len(lengths) - 1:
            del boundaries[-2]
        else:
            start, middle, end = boundaries[short_index : short_index + 3]
            short_mean = features[start:middle].mean(axis=0)
            left_mean = features[boundaries[short_index - 1] : start].mean(axis=0)
            right_mean = features[middle:end].mean(axis=0)
            if np.linalg.norm(short_mean - left_mean) <= np.linalg.norm(short_mean - right_mean):
                del boundaries[short_index]
            else:
                del boundaries[short_index + 1]
    return boundaries


def _split_long_segments(
    boundaries: list[int],
    features,
    minimum_frames: int,
    maximum_frames: int,
    split_threshold_frames: int,
) -> list[int]:
    import numpy as np

    result = [boundaries[0]]
    for original_start, original_end in zip(boundaries[:-1], boundaries[1:]):
        pending = [(original_start, original_end)]
        pieces = []
        while pending:
            start, end = pending.pop()
            if end - start <= split_threshold_frames:
                pieces.append((start, end))
                continue
            valid_start = start + minimum_frames
            valid_end = end - minimum_frames
            if valid_start >= valid_end:
                cut = start + (end - start) // 2
            else:
                # Compare sustained behavior on both sides of each candidate.
                # This deliberately ignores isolated one-frame spikes.
                comparison_window = max(2, minimum_frames // 2)
                candidates = range(valid_start, valid_end + 1)
                scores = [
                    np.linalg.norm(
                        features[max(start, point - comparison_window) : point].mean(axis=0)
                        - features[point : min(end, point + comparison_window)].mean(axis=0)
                    )
                    for point in candidates
                ]
                cut = valid_start + int(np.argmax(scores))
            pending.extend([(cut, end), (start, cut)])
        for _, end in sorted(pieces):
            result.append(end)
    return result


def _decode_annotations(store) -> list[dict]:
    import numpy as np

    if "annotations" not in store:
        return []
    decoded = []
    for raw in store["annotations"][:]:
        while isinstance(raw, np.ndarray):
            raw = raw.item() if raw.shape == () else raw.flat[0]
        if isinstance(raw, np.bytes_):
            raw = bytes(raw)
        if isinstance(raw, (bytes, bytearray, memoryview)):
            raw = bytes(raw).decode("utf-8")
        record = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(record, dict):
            decoded.append(record)
    return decoded


def _best_annotation(start: int, end: int, annotations: list[dict]):
    overlaps = []
    for annotation in annotations:
        annotation_start = int(annotation.get("start_idx", -1))
        annotation_end = int(annotation.get("end_idx", -1))
        overlap = max(0, min(end, annotation_end) - max(start, annotation_start))
        if overlap:
            overlaps.append((overlap, str(annotation.get("text", ""))))
    return max(overlaps, default=(0, None))[1]


def _make_segments(boundaries: list[int], timestamps, fps: float, annotations: list[dict]) -> list[dict]:
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start_time = float(timestamps[start])
        end_time = float(timestamps[end]) if end < len(timestamps) else float(timestamps[-1] + 1.0 / fps)
        segments.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "start_time": start_time,
                "end_time": end_time,
                "annotation": _best_annotation(start, end, annotations),
            }
        )
    return segments


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    cpu=4,
    memory=8192,
    timeout=60 * 60,
)
def segment_episode(
    episode_id: str,
    minimum_seconds: float = 1.5,
    maximum_seconds: float = 4.0,
    smoothing_seconds: float = 0.25,
    penalty: float = 120.0,
    head_weight: float = 0.0,
    mode: str = "both",
) -> dict:
    import numpy as np
    import ruptures as rpt
    import zarr

    if mode not in {"kinematic", "fixed", "both"}:
        raise ValueError("mode must be kinematic, fixed, or both")
    if not 0 < minimum_seconds <= maximum_seconds:
        raise ValueError("Require 0 < minimum_seconds <= maximum_seconds")

    volume.reload()
    store = zarr.open_group(str(EPISODES / episode_id), mode="r")
    fps = float(store.attrs.get("fps", 30.0))
    frame_count = min(
        int(store.attrs.get("total_frames", store["left.obs_ee_pose"].shape[0])),
        int(store["left.obs_ee_pose"].shape[0]),
        int(store["right.obs_ee_pose"].shape[0]),
    )
    annotations = _decode_annotations(store)
    destination = OUTPUT / episode_id
    results = {}

    if mode in {"kinematic", "both"}:
        smoothing_frames = max(1, round(smoothing_seconds * fps))
        features, timestamps, feature_metadata = _kinematic_features(
            store, frame_count, fps, smoothing_frames, head_weight
        )
        minimum_frames = max(2, round(minimum_seconds * fps))
        maximum_frames = max(minimum_frames, round(maximum_seconds * fps))
        # A small grace region avoids splitting an otherwise coherent phrase
        # merely because it exceeds the target maximum by a few frames.
        split_threshold_frames = max(maximum_frames, round(maximum_seconds * 1.15 * fps))
        detected = rpt.Pelt(model="l2", min_size=minimum_frames, jump=1).fit(features).predict(pen=penalty)
        boundaries = [0] + [int(point) for point in detected if 0 < point <= frame_count]
        if boundaries[-1] != frame_count:
            boundaries.append(frame_count)
        boundaries = _merge_short_segments(boundaries, features, minimum_frames)
        boundaries = _split_long_segments(
            boundaries,
            features,
            minimum_frames,
            maximum_frames,
            split_threshold_frames,
        )
        payload = {
            "episode": episode_id,
            "method": "kinematic_pelt",
            "fps": fps,
            "frame_count": frame_count,
            "parameters": {
                "minimum_seconds": minimum_seconds,
                "maximum_seconds": maximum_seconds,
                "smoothing_seconds": smoothing_seconds,
                "smoothing_frames": smoothing_frames,
                "penalty": penalty,
                "head_weight": head_weight,
                "maximum_split_tolerance": 0.15,
            },
            "kinematic_state": feature_metadata,
            "segments": _make_segments(boundaries, timestamps, fps, annotations),
        }
        _write_json_atomic(destination / "kinematic.json", payload)
        results["kinematic_segments"] = len(payload["segments"])
    else:
        timestamps, _ = _timestamps_seconds(store, frame_count, fps)

    if mode in {"fixed", "both"}:
        step = max(1, round(fps))
        fixed_boundaries = list(range(0, frame_count, step)) + [frame_count]
        fixed_payload = {
            "episode": episode_id,
            "method": "fixed_1s",
            "fps": fps,
            "frame_count": frame_count,
            "segments": _make_segments(fixed_boundaries, timestamps, fps, annotations),
        }
        _write_json_atomic(destination / "fixed_1s.json", fixed_payload)
        results["fixed_segments"] = len(fixed_payload["segments"])

    volume.commit()
    result = {"episode": episode_id, **results}
    print(json.dumps(result))
    return result


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 5)
def list_episode_ids(max_episodes: int) -> list[str]:
    volume.reload()
    episode_ids = sorted(
        path.name for path in EPISODES.iterdir() if path.is_dir() and (path / "zarr.json").is_file()
    )
    return episode_ids if max_episodes <= 0 else episode_ids[:max_episodes]


@app.local_entrypoint()
def main(
    max_episodes: int = 2,
    minimum_seconds: float = 1.5,
    maximum_seconds: float = 4.0,
    smoothing_seconds: float = 0.25,
    penalty: float = 120.0,
    head_weight: float = 0.0,
    mode: str = "both",
):
    episode_ids = list_episode_ids.remote(max_episodes)
    print(f"Computing {mode} boundaries for {len(episode_ids)} episode(s)")
    for result in segment_episode.map(
        episode_ids,
        kwargs={
            "minimum_seconds": minimum_seconds,
            "maximum_seconds": maximum_seconds,
            "smoothing_seconds": smoothing_seconds,
            "penalty": penalty,
            "head_weight": head_weight,
            "mode": mode,
        },
    ):
        print(result)
