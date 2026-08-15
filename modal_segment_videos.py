"""Convert EgoVerse JPEG frame arrays into one-second MP4 segments on Modal.

Smoke test one episode:
    modal run modal_segment_videos.py

Process more episodes:
    modal run modal_segment_videos.py --max-episodes 10
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal


APP_NAME = "egoverse-video-segmenter"
VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
EPISODES = MOUNT / "episodes"
SEGMENTS = MOUNT / "segments"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("zarr==3.1.5", "simplejpeg")
)


def _encode_segment(jpeg_frames, fps: float, output_path: Path) -> None:
    """Decode JPEGs and stream RGB frames into an H.264 MP4."""
    import simplejpeg

    first = simplejpeg.decode_jpeg(bytes(jpeg_frames[0]), colorspace="RGB")
    height, width = first.shape[:2]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        process.stdin.write(first.tobytes())
        for encoded in jpeg_frames[1:]:
            frame = simplejpeg.decode_jpeg(bytes(encoded), colorspace="RGB")
            if frame.shape[:2] != (height, width):
                raise ValueError("Frame dimensions changed within an episode")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(stderr.decode(errors="replace"))


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 6,
)
def segment_episode(
    episode_id: str,
    camera_key: str = "images.front_1",
    segment_seconds: float = 1.0,
    overwrite: bool = False,
) -> dict:
    import zarr

    volume.reload()
    source = EPISODES / episode_id
    store = zarr.open_group(str(source), mode="r")
    if camera_key not in store:
        raise KeyError(f"{episode_id} does not contain {camera_key}")

    frames = store[camera_key]
    fps = float(store.attrs.get("fps", 30))
    frames_per_segment = max(1, round(fps * segment_seconds))
    stored_frame_count = int(frames.shape[0])
    total_frames = min(
        int(store.attrs.get("total_frames", stored_frame_count)), stored_frame_count
    )
    camera_name = camera_key.removeprefix("images.")
    destination = SEGMENTS / episode_id / camera_name
    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = destination / "manifest.json"
    expected_count = math.ceil(total_frames / frames_per_segment)
    if manifest_path.exists() and not overwrite:
        previous = json.loads(manifest_path.read_text())
        expected_files_exist = all(
            (destination / f"{index:06d}.mp4").is_file()
            and (destination / f"{index:06d}.mp4").stat().st_size > 0
            for index in range(expected_count)
        )
        if (
            previous.get("segment_count") == expected_count
            and previous.get("camera_key") == camera_key
            and previous.get("requested_segment_seconds") == segment_seconds
            and previous.get("fps") == fps
            and expected_files_exist
        ):
            return {"episode": episode_id, "status": "already_complete", **previous}

    manifest_segments = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        for index, start in enumerate(range(0, total_frames, frames_per_segment)):
            end = min(start + frames_per_segment, total_frames)
            final_path = destination / f"{index:06d}.mp4"
            if not (final_path.exists() and final_path.stat().st_size > 0 and not overwrite):
                temporary_path = temporary / f"{index:06d}.mp4"
                _encode_segment(frames[start:end], fps, temporary_path)
                partial_path = destination / f"{index:06d}.mp4.partial"
                shutil.copyfile(temporary_path, partial_path)
                partial_path.replace(final_path)
            manifest_segments.append(
                {
                    "file": final_path.name,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "start_seconds": start / fps,
                    "duration_seconds": (end - start) / fps,
                }
            )

    manifest = {
        "episode": episode_id,
        "camera_key": camera_key,
        "fps": fps,
        "total_frames": total_frames,
        "frames_per_segment": frames_per_segment,
        "requested_segment_seconds": segment_seconds,
        "segment_count": len(manifest_segments),
        "segments": manifest_segments,
    }
    temporary_manifest = destination / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    volume.commit()
    result = {"status": "created", **manifest}
    print(json.dumps({key: value for key, value in result.items() if key != "segments"}))
    return result


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 5)
def list_episode_ids(max_episodes: int) -> list[str]:
    volume.reload()
    episode_ids = sorted(
        path.name
        for path in EPISODES.iterdir()
        if path.is_dir() and (path / "zarr.json").is_file()
    )
    return episode_ids if max_episodes <= 0 else episode_ids[:max_episodes]


@app.local_entrypoint()
def main(
    max_episodes: int = 1,
    camera_key: str = "images.front_1",
    segment_seconds: float = 1.0,
    overwrite: bool = False,
):
    episode_ids = list_episode_ids.remote(max_episodes)
    print(f"Segmenting {len(episode_ids)} episode(s) into {segment_seconds}s clips")
    for result in segment_episode.map(
        episode_ids,
        kwargs={
            "camera_key": camera_key,
            "segment_seconds": segment_seconds,
            "overwrite": overwrite,
        },
    ):
        print(result["episode"], result["status"], result["segment_count"])
