"""Fetch real camera frames for each dashboard clip and encode them as small MP4s.

The kinematics sync deliberately excluded `images.front_1` (video is ~97% of the
bytes). But mecka episodes store one JPEG per chunk, so the frames backing a
single 10-second cycle can be pulled without touching the rest of the episode.

Each clip becomes a short, heavily-compressed MP4 inlined into the dashboard as
a data URI -- the artifact CSP blocks external hosts, so nothing can be streamed.

    modal run modal_clip_videos.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import modal

VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
EPISODES = MOUNT / "episodes"
META = MOUNT / "metadata"
RESULTS = MOUNT / "derived" / "results"
BUCKET = "rldb"

# Playback budget per clip. Long cycles are sampled, not truncated, so the whole
# action is still visible -- just faster.
MAX_FRAMES = 48
OUT_FPS = 12
OUT_WIDTH = 288

app = modal.App("egoverse-clip-videos")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("zarr==3.1.5", "simplejpeg", "boto3", "numpy==2.2.6",
                 "pandas==2.3.3", "pyarrow==21.0.0")
)
r2 = modal.Secret.from_dotenv(path="/private/tmp", filename="egoverse_env")


def _client():
    import os

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", max_pool_connections=32,
                      retries={"max_attempts": 4, "mode": "standard"}),
    )


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=900)
def plan() -> list[dict]:
    """Resolve each clip to a full episode hash, R2 prefix and frame range."""
    import pandas as pd

    volume.reload()
    clips = json.loads((RESULTS / "clips.json").read_text())["clips"]
    meta = pd.read_parquet(META / "episodes.parquet")
    seg = pd.read_parquet(RESULTS / "segments_scored.parquet")

    # clips.json stores a truncated hash for display; recover the full one.
    by_prefix = {h[:10]: h for h in seg.episode_hash.unique()}
    paths = dict(zip(meta.episode_hash, meta.zarr_processed_path.fillna("")))

    out = []
    for i, c in enumerate(clips):
        full = by_prefix.get(c["episode"])
        if not full:
            continue
        p = (paths.get(full) or "").strip()
        if not p.startswith("s3://rldb/"):
            continue
        out.append(
            {
                "i": i,
                "verb": c["verb"],
                "kind": c["kind"],
                "episode": c["episode"],
                "prefix": p[len("s3://rldb/"):].rstrip("/"),
                "start": c["start_seconds"] if "start_seconds" in c else None,
                "duration": c["duration"],
            }
        )
    # start/end come from the scored segments, matched on episode + duration
    lookup = {}
    for r in seg.itertuples():
        lookup.setdefault(r.episode_hash[:10], []).append(
            (float(r.start_seconds), float(r.end_seconds), float(r.duration))
        )
    for o in out:
        cands = lookup.get(o["episode"], [])
        best = min(cands, key=lambda t: abs(t[2] - o["duration"]), default=None)
        if best:
            o["start"], o["end"] = best[0], best[1]
    out = [o for o in out if o.get("start") is not None]
    print(f"planned {len(out)} of {len(clips)} clips")
    return out


@app.function(
    image=image, secrets=[r2], volumes={str(MOUNT): volume},
    timeout=1800, cpu=2, memory=8192, max_containers=25,
    retries=modal.Retries(max_retries=2, initial_delay=2.0),
)
def render(job: dict) -> dict:
    """Pull just this clip's JPEG chunks from R2 and encode a small MP4."""
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    import simplejpeg
    import zarr

    s3 = _client()
    prefix = job["prefix"]
    tmp = Path(tempfile.mkdtemp())
    store = tmp / "ep.zarr"
    (store / "images.front_1").mkdir(parents=True, exist_ok=True)

    def fetch(key: str, dest: Path) -> bool:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, f"{prefix}/{key}", str(dest))
            return True
        except Exception:
            return False

    if not fetch("zarr.json", store / "zarr.json"):
        return {"i": job["i"], "error": "no root zarr.json"}
    if not fetch("images.front_1/zarr.json", store / "images.front_1" / "zarr.json"):
        return {"i": job["i"], "error": "no image zarr.json"}

    attrs = json.loads((store / "zarr.json").read_text())["attributes"]
    fps = float(attrs.get("fps", 30))
    total = int(attrs.get("total_frames", 0))
    a = max(0, int(round(job["start"] * fps)))
    b = min(total, int(round(job["end"] * fps)))
    if b - a < 2:
        return {"i": job["i"], "error": "range too short"}

    idx = np.unique(np.linspace(a, b - 1, min(MAX_FRAMES, b - a)).round().astype(int))
    got = [i for i in idx if fetch(f"images.front_1/c/{i}", store / "images.front_1" / "c" / str(i))]
    if len(got) < 2:
        return {"i": job["i"], "error": f"only {len(got)} chunks"}

    # Elements are raw JPEG byte blobs, not decoded pixels -- indexing yields a
    # 0-d object array, so each frame has to go through the JPEG decoder.
    arr = zarr.open_group(str(store), mode="r")["images.front_1"]
    frames = []
    for i in got:
        try:
            frames.append(simplejpeg.decode_jpeg(bytes(arr[i]), colorspace="RGB"))
        except Exception as exc:  # noqa: BLE001
            if not frames:
                print(f"  decode fail {job['verb']}: {type(exc).__name__}: {exc}")
            continue
    if len(frames) < 2:
        return {"i": job["i"], "error": "decode failed"}
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        keep = frames[0].shape
        frames = [f for f in frames if f.shape == keep]

    h, w = frames[0].shape[:2]
    out_w = OUT_WIDTH - (OUT_WIDTH % 2)
    out_h = int(round(h * out_w / w))
    out_h -= out_h % 2
    mp4 = tmp / "clip.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{w}x{h}",
        "-r", str(OUT_FPS), "-i", "pipe:0", "-an",
        "-vf", f"scale={out_w}:{out_h}",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "33",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        err = proc.stderr.read()
        if proc.wait():
            return {"i": job["i"], "error": err.decode(errors="replace")[:200]}
    except BrokenPipeError:
        return {"i": job["i"], "error": "ffmpeg pipe closed"}

    data = mp4.read_bytes()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [{job['i']}] {job['verb']}/{job['kind']} {len(frames)}f -> {len(data)/1024:.0f}KB")
    return {"i": job["i"], "frames": len(frames), "bytes": len(data),
            "b64": base64.b64encode(data).decode()}


@app.local_entrypoint()
def main():
    jobs = plan.remote()
    results = [r for r in render.map(jobs, order_outputs=False) if r]
    ok = [r for r in results if "b64" in r]
    bad = [r for r in results if "b64" not in r]
    videos = {str(r["i"]): r["b64"] for r in ok}
    total = sum(r["bytes"] for r in ok)
    print(f"\nencoded {len(ok)}/{len(jobs)}  total {total/1e6:.2f} MB  "
          f"mean {total/max(len(ok),1)/1024:.0f} KB")
    for r in bad[:10]:
        print("  FAIL", r)
    Path("scratch_results/clip_videos.json").write_text(json.dumps(videos))
    print("wrote scratch_results/clip_videos.json")
