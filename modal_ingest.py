"""Bounded ingestion of existing EgoVerse Zarr episodes into a Modal Volume.

This intentionally copies Zarr stores byte-for-byte. It does not decode video,
rewrite arrays, or depend on EgoVerse's private PostgreSQL episode registry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "egoverse-ingest"
VOLUME_NAME = "egoverse-zarrs"
MOUNT_PATH = Path("/egoverse")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "boto3~=1.40",
    "zarr==3.1.5",
)

r2_secret = modal.Secret.from_name(
    "egoverse-r2",
    required_keys=[
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_ENDPOINT_URL",
    ],
)


def _r2_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _episode_root(key: str, source_prefix: str) -> str | None:
    """Return the first *.zarr directory underneath source_prefix."""
    relative = key.removeprefix(source_prefix).lstrip("/")
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        if part.endswith(".zarr"):
            return "/".join(parts[: index + 1])
    return None


@app.function(
    image=image,
    secrets=[r2_secret],
    volumes={str(MOUNT_PATH): volume},
    timeout=60 * 60 * 6,
    ephemeral_disk=1024,
)
def ingest(
    source_prefix: str,
    max_episodes: int = 1,
    bucket: str = "rldb",
) -> dict:
    """Copy at most max_episodes complete Zarr stores into the Modal Volume."""
    if max_episodes < 1:
        raise ValueError("max_episodes must be at least 1")

    source_prefix = source_prefix.strip("/") + "/"
    s3 = _r2_client()
    paginator = s3.get_paginator("list_objects_v2")

    episode_roots: list[str] = []
    seen: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=source_prefix):
        for obj in page.get("Contents", []):
            root = _episode_root(obj["Key"], source_prefix)
            if root and root not in seen:
                seen.add(root)
                episode_roots.append(root)
                if len(episode_roots) >= max_episodes:
                    break
        if len(episode_roots) >= max_episodes:
            break

    if not episode_roots:
        raise RuntimeError(
            f"No .zarr stores found at s3://{bucket}/{source_prefix}"
        )

    copied_files = 0
    copied_bytes = 0
    episodes: list[dict] = []

    for episode_root in episode_roots:
        remote_episode_prefix = source_prefix + episode_root.rstrip("/") + "/"
        local_episode = MOUNT_PATH / "episodes" / episode_root
        local_episode.mkdir(parents=True, exist_ok=True)

        episode_files = 0
        episode_bytes = 0
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=remote_episode_prefix,
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                relative = key.removeprefix(remote_episode_prefix)
                destination = local_episode / relative
                destination.parent.mkdir(parents=True, exist_ok=True)

                size = int(obj["Size"])
                if not destination.exists() or destination.stat().st_size != size:
                    s3.download_file(bucket, key, str(destination))

                episode_files += 1
                episode_bytes += size

        metadata_path = local_episode / "zarr.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Incomplete Zarr store: {metadata_path} is missing")

        metadata = json.loads(metadata_path.read_text())
        attrs = metadata.get("attributes", {})
        episodes.append(
            {
                "episode": episode_root,
                "files": episode_files,
                "bytes": episode_bytes,
                "total_frames": attrs.get("total_frames"),
                "task_name": attrs.get("task_name"),
                "embodiment": attrs.get("embodiment"),
            }
        )
        copied_files += episode_files
        copied_bytes += episode_bytes

    # Make the completed stores visible to subsequent Modal functions.
    volume.commit()
    return {
        "bucket": bucket,
        "source_prefix": source_prefix,
        "volume": VOLUME_NAME,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "episodes": episodes,
    }


@app.function(image=image, volumes={str(MOUNT_PATH): volume})
def inventory() -> list[dict]:
    """Read the root metadata for all ingested episodes."""
    volume.reload()
    results = []
    root = MOUNT_PATH / "episodes"
    if not root.exists():
        return results

    for metadata_path in sorted(root.rglob("*.zarr/zarr.json")):
        metadata = json.loads(metadata_path.read_text())
        attrs = metadata.get("attributes", {})
        results.append(
            {
                "path": str(metadata_path.parent.relative_to(MOUNT_PATH)),
                "total_frames": attrs.get("total_frames"),
                "fps": attrs.get("fps"),
                "task_name": attrs.get("task_name"),
                "embodiment": attrs.get("embodiment"),
                "feature_keys": sorted(attrs.get("features", {})),
            }
        )
    return results


@app.local_entrypoint()
def main(
    source_prefix: str = "",
    max_episodes: int = 1,
    bucket: str = "rldb",
    list_only: bool = False,
):
    if list_only:
        print(json.dumps(inventory.remote(), indent=2))
        return
    if not source_prefix:
        raise ValueError("Pass --source-prefix for a bounded ingestion")
    result = ingest.remote(source_prefix, max_episodes, bucket)
    print(json.dumps(result, indent=2))
