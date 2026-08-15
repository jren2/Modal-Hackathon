"""Extract complete task attempts from raw EgoVerse episodes.

Each one-second window is classified as TASK, RESET, or IRRELEVANT using an
open-source VLM, then adjusted with annotation and hand-activity evidence.
The windows are temporally cleaned and contiguous TASK regions become attempts.

Run the first episode for visual validation:
    modal run modal_extract_attempts.py --max-episodes 1
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import modal


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PIPELINE_VERSION = "attempt-v1-fast-scene-gate"
VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
EPISODES = MOUNT / "episodes"
OUTPUT = MOUNT / "attempts"

app = modal.App("egoverse-attempt-extraction")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
_MODEL = None
_PROCESSOR = None
_MODEL_LOCK = threading.Lock()


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.9.0",
        "huggingface-hub==0.34.3",
        "numpy==2.2.6",
        "pillow==11.3.0",
        "qwen-vl-utils==0.0.11",
        "scipy==1.15.3",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==4.54.1",
        "zarr==3.1.5",
        "simplejpeg==1.9.0",
    )
    .run_commands(
        "huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /models/qwen-vl",
    )
    .run_commands("test -f /models/qwen-vl/config.json")
)
orchestrator_image = modal.Image.debian_slim(python_version="3.11")


def _decode_json_value(raw):
    import numpy as np

    while isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.flat[0]
    if isinstance(raw, np.bytes_):
        raw = bytes(raw)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw).decode("utf-8")
    return json.loads(raw) if isinstance(raw, str) else raw


def _jpeg_bytes(raw) -> bytes:
    import numpy as np

    while isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.flat[0]
    if isinstance(raw, np.bytes_):
        raw = bytes(raw)
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"Unexpected JPEG payload type: {type(raw)!r}")
    return bytes(raw)


def _annotations(store) -> list[dict]:
    if "annotations" not in store:
        return []
    return [record for raw in store["annotations"][:] if isinstance((record := _decode_json_value(raw)), dict)]


def _overlapping_annotations(start: int, end: int, annotations: list[dict]) -> list[str]:
    return [
        str(annotation.get("text", ""))
        for annotation in annotations
        if max(start, int(annotation.get("start_idx", -1)))
        < min(end, int(annotation.get("end_idx", -1)))
    ]


def _annotation_evidence(texts: list[str]) -> tuple[float, bool]:
    if not texts:
        return 0.0, False
    combined = " ".join(texts).lower()
    reset_words = {
        "reset", "prepare", "preparing", "unfold", "unfolding", "restore",
        "starting position", "wait", "waiting", "setup", "clean", "cleanup",
    }
    reset = any(word in combined for word in reset_words)
    return (0.25 if reset else 0.85), reset


def _timestamps(store, frame_count: int, fps: float):
    import numpy as np

    if "obs_rgb_timestamps_ns" in store:
        raw = np.asarray(store["obs_rgb_timestamps_ns"][:frame_count], dtype=np.float64).reshape(-1)
        if len(raw) == frame_count and np.all(np.diff(raw) > 0):
            seconds = (raw - raw[0]) * 1e-9
            delta = np.diff(seconds)
            nominal = 1.0 / fps
            valid = (delta >= nominal * 0.25) & (delta <= nominal * 4.0)
            replacement = np.median(delta[valid]) if np.any(valid) else nominal
            return np.concatenate(([0.0], np.cumsum(np.where(valid, delta, replacement))))
    return np.arange(frame_count, dtype=np.float64) / fps


def _hand_activity(store, frame_count: int, fps: float):
    import numpy as np

    left = np.asarray(store["left.obs_ee_pose"][:frame_count, :3], dtype=np.float64)
    right = np.asarray(store["right.obs_ee_pose"][:frame_count, :3], dtype=np.float64)
    if "obs_head_pose" in store:
        from scipy.spatial.transform import Rotation

        head = np.asarray(store["obs_head_pose"][:frame_count], dtype=np.float64)
        quaternion = head[:, 3:7]
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        inverse_head = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).inv()
        left = inverse_head.apply(left - head[:, :3])
        right = inverse_head.apply(right - head[:, :3])
    speed = np.concatenate(
        ([0.0], (np.linalg.norm(np.diff(left, axis=0), axis=1) + np.linalg.norm(np.diff(right, axis=0), axis=1)) * fps)
    )
    return speed


def _parse_vlm_response(text: str) -> tuple[str, float, str]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else text)
        label = str(payload.get("class", "")).upper()
        confidence = float(payload.get("confidence", 0.7))
        reason = str(payload.get("reason", ""))
        if label in {"TASK", "RESET", "IRRELEVANT"}:
            return label, min(1.0, max(0.0, confidence)), reason
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    upper = text.upper()
    for label in ("IRRELEVANT", "RESET", "TASK"):
        if label in upper:
            return label, 0.6, text[:200]
    return "IRRELEVANT", 0.5, "Unparseable VLM response"


def _parse_scene_gate(text: str) -> tuple[bool, float, str]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else text)
        decision = str(payload.get("decision", "")).strip().upper()
        if decision not in {"VALID", "INVALID"}:
            raise ValueError(f"Unexpected scene decision: {decision!r}")
        confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.7))))
        valid = decision == "VALID" and confidence >= 0.5
        return valid, confidence, str(payload.get("reason", ""))
    except (ValueError, TypeError, json.JSONDecodeError):
        # Fail closed: uncertain/unparseable scenes should not become training attempts.
        return False, 0.5, f"Unparseable scene-gate response: {text[:160]}"


def _combine_evidence(visual_class, visual_confidence, annotation_relevance, annotation_reset, activity):
    labels = ("TASK", "RESET", "IRRELEVANT")
    scores = {label: (1.0 - visual_confidence) / 2.0 for label in labels}
    scores[visual_class] = visual_confidence
    scores["TASK"] += 0.18 * annotation_relevance + 0.12 * activity
    if annotation_reset:
        scores["RESET"] += 0.25
        scores["TASK"] -= 0.10
    total = sum(max(0.0, value) for value in scores.values())
    probabilities = {key: max(0.0, value) / total for key, value in scores.items()}
    final = max(probabilities, key=probabilities.get)
    return final, probabilities[final]


def _runs(labels: list[str]):
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            yield start, index, labels[start]
            start = index


def _temporal_cleanup(labels: list[str], fill_gap_windows: int, minimum_task_windows: int) -> list[str]:
    cleaned = list(labels)
    # Fill a very short non-task gap only when TASK exists on both sides.
    for start, end, label in list(_runs(cleaned)):
        if label == "RESET" and end - start <= fill_gap_windows and start > 0 and end < len(cleaned):
            if cleaned[start - 1] == cleaned[end] == "TASK":
                cleaned[start:end] = ["TASK"] * (end - start)
    # Remove isolated/unsustained TASK predictions.
    for start, end, label in list(_runs(cleaned)):
        if label == "TASK" and end - start < minimum_task_windows:
            left = cleaned[start - 1] if start else None
            right = cleaned[end] if end < len(cleaned) else None
            replacement = left if left == right and left else (left or right or "IRRELEVANT")
            cleaned[start:end] = [replacement] * (end - start)
    return cleaned


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    volumes={str(MOUNT): volume},
    timeout=60 * 60,
    max_containers=32,
)
def extract_attempts(
    episode_id: str,
    window_seconds: float = 1.0,
    fill_gap_windows: int = 1,
    minimum_task_windows: int = 3,
) -> dict:
    import numpy as np
    import simplejpeg
    import torch
    import zarr
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    volume.reload()
    store = zarr.open_group(str(EPISODES / episode_id), mode="r")
    task = str(store.attrs.get("task_description") or store.attrs.get("task_name") or "unknown task")
    fps = float(store.attrs.get("fps", 30.0))
    frame_count = min(int(store.attrs["total_frames"]), int(store["images.front_1"].shape[0]))
    step = max(1, round(window_seconds * fps))
    timestamps = _timestamps(store, frame_count, fps)
    activity_per_frame = _hand_activity(store, frame_count, fps)
    annotations = _annotations(store)

    destination = OUTPUT / episode_id / "attempts.json"
    requested_parameters = {
        "window_seconds": window_seconds,
        "fill_gap_windows": fill_gap_windows,
        "minimum_task_windows": minimum_task_windows,
    }
    if destination.is_file():
        previous = json.loads(destination.read_text())
        if (
            previous.get("pipeline_version") == PIPELINE_VERSION
            and previous.get("parameters") == requested_parameters
        ):
            return {
                "episode_id": episode_id,
                "attempt_count": len(previous.get("attempts", [])),
                "status": "already_complete",
            }

    global _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is None or _PROCESSOR is None:
            loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/models/qwen-vl", torch_dtype=torch.bfloat16, device_map="auto"
            )
            loaded_model.generation_config.temperature = None
            loaded_processor = AutoProcessor.from_pretrained("/models/qwen-vl")
            _MODEL = loaded_model
            _PROCESSOR = loaded_processor
    model = _MODEL
    processor = _PROCESSOR

    windows = []
    raw_activity = []
    for start in range(0, frame_count, step):
        end = min(start + step, frame_count)
        raw_activity.append(float(np.median(activity_per_frame[start:end])))
    low, high = np.percentile(raw_activity, [10, 90])

    for window_index, start in enumerate(range(0, frame_count, step)):
        end = min(start + step, frame_count)
        sample_indices = [start + (end - start) // 2]
        image_values = []
        for sample_index in sample_indices:
            rgb = simplejpeg.decode_jpeg(
                _jpeg_bytes(store["images.front_1"][sample_index]), colorspace="RGB"
            )
            image_value = Image.fromarray(rgb)
            image_value.thumbnail((448, 448))
            image_values.append(image_value)
        annotation_texts = _overlapping_annotations(start, end, annotations)
        annotation_relevance, annotation_reset = _annotation_evidence(annotation_texts)
        center_image = image_values[len(image_values) // 2]
        gate_prompt = f"""Inspect only what is visibly present in this image.
The expected activity is: {task}.
A valid task scene requires the relevant task object (for folding clothes: a clearly visible shirt, garment, towel, or cloth) and a usable first-person work area.
Mark the scene INVALID when the task object is absent or hidden, the view is dominated/blocked by another person's face or torso, the image shows an empty room/table, researchers are setting up, the camera is being adjusted, or cleanup/unrelated activity is occurring.
Do not infer an object that is not visibly present. Hands or motion alone are not enough.
Return only a JSON object with these fields:
- "decision": exactly "VALID" or "INVALID"
- "confidence": a number from 0.5 to 1.0
- "reason": the specific objects/people visibly present or absent
Do not copy the field descriptions as the answer."""
        gate_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": center_image},
                    {"type": "text", "text": gate_prompt},
                ],
            }
        ]
        gate_inputs = processor.apply_chat_template(
            gate_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            gate_generated = model.generate(
                **gate_inputs, max_new_tokens=100, do_sample=False
            )
        gate_response = processor.batch_decode(
            gate_generated[:, gate_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        scene_valid, scene_confidence, scene_reason = _parse_scene_gate(gate_response)

        if scene_valid:
            visual_class = "RESET" if annotation_reset else "TASK"
            visual_confidence = scene_confidence
            visual_reason = f"Single-pass scene gate: {scene_reason}"
        else:
            visual_class = "IRRELEVANT"
            visual_confidence = scene_confidence
            visual_reason = f"Scene gate: {scene_reason}"
        hand_activity = float(np.clip((raw_activity[window_index] - low) / max(high - low, 1e-8), 0, 1))
        final_class, confidence = _combine_evidence(
            visual_class, visual_confidence, annotation_relevance, annotation_reset, hand_activity
        )
        windows.append(
            {
                "window_index": window_index,
                "start_idx": start,
                "end_idx": end,
                "start_sec": float(timestamps[start]),
                "end_sec": float(timestamps[end]) if end < frame_count else float(timestamps[-1] + 1 / fps),
                "sample_frame_idxs": sample_indices,
                "scene_valid": scene_valid,
                "scene_confidence": scene_confidence,
                "scene_reason": scene_reason,
                "scene_raw_response": gate_response,
                "visual_class": visual_class,
                "visual_confidence": visual_confidence,
                "visual_reason": visual_reason,
                "annotations": annotation_texts,
                "annotation_relevance": annotation_relevance,
                "hand_activity": hand_activity,
                "raw_final_class": final_class,
                "confidence": confidence,
            }
        )

    cleaned = _temporal_cleanup(
        [window["raw_final_class"] for window in windows], fill_gap_windows, minimum_task_windows
    )
    for window, label in zip(windows, cleaned):
        window["final_class"] = label

    attempts = []
    for start_window, end_window, label in _runs(cleaned):
        if label != "TASK":
            continue
        selected = windows[start_window:end_window]
        attempts.append(
            {
                "attempt_id": len(attempts),
                "start_idx": selected[0]["start_idx"],
                "end_idx": selected[-1]["end_idx"],
                "start_sec": selected[0]["start_sec"],
                "end_sec": selected[-1]["end_sec"],
                "confidence": float(np.mean([window["confidence"] for window in selected])),
            }
        )

    visual_counts = {
        label: sum(window["visual_class"] == label for window in windows)
        for label in ("TASK", "RESET", "IRRELEVANT")
    }
    warnings = []
    dominant_label, dominant_count = max(visual_counts.items(), key=lambda item: item[1])
    if dominant_count / max(1, len(windows)) >= 0.95:
        warnings.append(
            f"visual_classifier_collapsed_to_{dominant_label}: human review required"
        )
    invalid_fraction = sum(not window["scene_valid"] for window in windows) / max(
        1, len(windows)
    )
    if invalid_fraction >= 0.9:
        warnings.append(
            "scene_gate_rejected_at_least_90_percent: human review required"
        )
    if not attempts:
        warnings.append("no_attempts_extracted: human review required")

    payload = {
        "episode_id": episode_id,
        "pipeline_version": PIPELINE_VERSION,
        "task": task,
        "model": MODEL_ID,
        "parameters": requested_parameters,
        "needs_human_review": bool(warnings),
        "warnings": warnings,
        "visual_class_counts": visual_counts,
        "attempts": attempts,
        "windows": windows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(destination)
    volume.commit()
    result = {
        "episode_id": episode_id,
        "task": task,
        "attempt_count": len(attempts),
        "needs_human_review": bool(warnings),
        "warnings": warnings,
    }
    print(json.dumps(result))
    return result


@app.function(
    image=orchestrator_image,
    volumes={str(MOUNT): volume},
    cpu=2,
    memory=4096,
    timeout=60 * 60 * 24,
)
def run_all(
    window_seconds: float = 1.0,
    fill_gap_windows: int = 1,
    minimum_task_windows: int = 3,
) -> dict:
    volume.reload()
    episode_ids = sorted(
        path.name
        for path in EPISODES.iterdir()
        if path.is_dir() and (path / "zarr.json").is_file()
    )
    print(f"Discovered {len(episode_ids)} episodes; submitting GPU calls")
    calls = []
    for index, episode_id in enumerate(episode_ids, start=1):
        calls.append(extract_attempts.spawn(
            episode_id,
            window_seconds,
            fill_gap_windows,
            minimum_task_windows,
        ))
        if index % 50 == 0 or index == len(episode_ids):
            print(f"Submitted {index}/{len(episode_ids)} calls")
    results = []
    failures = []
    for episode_id, call in zip(episode_ids, calls):
        try:
            results.append(call.get())
        except Exception as error:
            failures.append({"episode_id": episode_id, "error": str(error)})
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "episode_count": len(episode_ids),
        "completed": len(results),
        "failed": len(failures),
        "failures": failures,
    }
    summary_path = OUTPUT / "_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    print(json.dumps(summary))
    return summary


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 5)
def list_episode_ids(max_episodes: int) -> list[str]:
    volume.reload()
    episode_ids = sorted(
        path.name for path in EPISODES.iterdir() if path.is_dir() and (path / "zarr.json").is_file()
    )
    return episode_ids if max_episodes <= 0 else episode_ids[:max_episodes]


@app.local_entrypoint()
def main(
    max_episodes: int = 1,
    window_seconds: float = 1.0,
    fill_gap_windows: int = 1,
    minimum_task_windows: int = 3,
):
    episode_ids = list_episode_ids.remote(max_episodes)
    print(f"Extracting attempts from {len(episode_ids)} episode(s)")
    # Run sequentially in V1 so one warm GPU container/model serves all episodes.
    for episode_id in episode_ids:
        print(
            extract_attempts.remote(
                episode_id, window_seconds, fill_gap_windows, minimum_task_windows
            )
        )
