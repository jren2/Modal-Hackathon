"""Inspect EgoVerse data stored in a Modal Volume.

Run locally with:
    modal run modal_inspect_volume.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import modal


app = modal.App("egoverse-volume-inspector")
volume = modal.Volume.from_name("egoverse-zarrs-v2", version=2)
image = modal.Image.debian_slim(python_version="3.11").pip_install("zarr==3.1.5")


@app.function(image=image, volumes={"/egoverse": volume}, timeout=60 * 5)
def probe_volume() -> dict:
    """Read one Zarr metadata file and one chunk to prove container access."""
    root = Path("/egoverse/episodes")
    volume.reload()

    episode = next(
        path for path in root.iterdir() if path.is_dir() and (path / "zarr.json").is_file()
    )
    metadata = episode / "zarr.json"
    metadata_preview = metadata.read_text()[:160]

    # Zarr v3 stores array chunks beneath directories named `c`.
    chunk = next(
        path
        for path in episode.rglob("*")
        if path.is_file() and "c" in path.relative_to(episode).parts
    )
    chunk_preview = chunk.read_bytes()[:16].hex()

    result = {
        "episode": episode.name,
        "metadata_path": str(metadata),
        "metadata_preview": metadata_preview,
        "chunk_path": str(chunk),
        "chunk_size_bytes": chunk.stat().st_size,
        "first_16_chunk_bytes_hex": chunk_preview,
    }
    print(result)
    return result


@app.function(image=image, volumes={"/egoverse": volume}, timeout=60 * 30)
def inspect_volume(sample_count: int = 10) -> dict:
    root = Path("/egoverse/episodes")
    if not root.exists():
        raise FileNotFoundError(f"Volume path does not exist: {root}")

    # Refresh the container's view in case another job recently committed data.
    volume.reload()

    file_count = 0
    total_bytes = 0
    extensions: Counter[str] = Counter()
    sample_files: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        extensions[path.suffix or "<no extension>"] += 1
        if len(sample_files) < sample_count:
            sample_files.append(str(path.relative_to(root)))

    # Each immediate child containing zarr.json is one downloaded Zarr episode.
    episodes = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "zarr.json").is_file()
    )

    result = {
        "volume": "egoverse-zarrs-v2",
        "mounted_at": "/egoverse",
        "episode_count": len(episodes),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 3),
        "extensions": dict(extensions.most_common()),
        "sample_episodes": [path.name for path in episodes[:sample_count]],
        "sample_files": sample_files,
    }
    print(result)
    return result


@app.local_entrypoint()
def main(sample_count: int = 10, full_count: bool = False):
    if full_count:
        inspect_volume.remote(sample_count)
    else:
        probe_volume.remote()
