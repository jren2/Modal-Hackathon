#!/usr/bin/env python3
"""Select and download a small, deterministic EgoVerse task subset.

Discovery uses EgoVerse's SQL helpers and ``DatasetFilter``. Transfer uses
``S3EpisodeResolver.sync_from_filters`` so storage paths and R2 behavior remain
owned by the official repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


DEFAULT_MAX_EPISODES = 10
DEFAULT_SEED = 42
ALL_TASK_SENTINELS = frozenset({"*", "all", "__all__"})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "egoverse_subset"


class DatasetFilterType(Protocol):
    def __call__(self, filter_lambdas: Sequence[str] | None = None) -> Any: ...


class ResolverType(Protocol):
    @classmethod
    def sync_from_filters(cls, **kwargs: Any) -> list[tuple[str, str]]: ...


@dataclass(frozen=True)
class EgoVerseApi:
    create_default_engine: Any
    episode_table_to_df: Any
    DatasetFilter: DatasetFilterType
    S3EpisodeResolver: ResolverType


@dataclass(frozen=True)
class EpisodeChoice:
    episode_id: str
    task: str
    lab: str
    scene: str
    demonstrator: str
    zarr_processed_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a bounded EgoVerse task subset from SQL metadata, then "
            "download only those complete Zarr episodes."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Exact EgoVerse task name. '*'/'all' requires "
            "--confirm-complete-dataset."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=DEFAULT_MAX_EPISODES,
        help=(
            f"Maximum episodes to select (default: {DEFAULT_MAX_EPISODES}). "
            "Use 0 for unlimited only with --confirm-complete-dataset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic selection seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and print the selection without downloading or writing files.",
    )
    parser.add_argument(
        "--confirm-complete-dataset",
        action="store_true",
        help="Explicitly permit an all-task or unbounded request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="s5cmd worker count passed to the official resolver (default: 10).",
    )
    parser.add_argument(
        "--bucket",
        default="rldb",
        help="Official EgoVerse bucket name (default: rldb).",
    )
    parser.add_argument(
        "--egoverse-repo",
        type=Path,
        default=None,
        help=(
            "Path to a GaTech-RL2/EgoVerse checkout when egomimic is not "
            "already installed. May also be set with EGOVERSE_REPO."
        ),
    )
    return parser


def validate_safety_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    task = args.task.strip()
    if not task:
        parser.error("--task must not be empty")
    args.task = task

    if args.max_episodes < 0:
        parser.error("--max-episodes must be 0 or greater")
    if args.max_episodes == 0 and not args.confirm_complete_dataset:
        parser.error(
            "an unlimited request (--max-episodes 0) requires "
            "--confirm-complete-dataset"
        )
    if task.casefold() in ALL_TASK_SENTINELS and not args.confirm_complete_dataset:
        parser.error(
            "an all-task request requires --confirm-complete-dataset; "
            "use an exact task name for a bounded subset"
        )
    if args.workers < 1:
        parser.error("--workers must be at least 1")


def load_official_api(repo_path: Path | None = None) -> EgoVerseApi:
    configured_path = repo_path
    if configured_path is None and os.environ.get("EGOVERSE_REPO"):
        configured_path = Path(os.environ["EGOVERSE_REPO"])

    if configured_path is not None:
        configured_path = configured_path.expanduser().resolve()
        if not (configured_path / "egomimic").is_dir():
            raise RuntimeError(
                f"Not an EgoVerse checkout (missing egomimic/): {configured_path}"
            )
        sys.path.insert(0, str(configured_path))

    try:
        from egomimic.rldb.filters import DatasetFilter
        from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
        from egomimic.utils.aws.aws_sql import (
            create_default_engine,
            episode_table_to_df,
        )
    except ImportError as exc:
        hint = (
            "Install the official GaTech-RL2/EgoVerse checkout and its "
            "dependencies, or pass --egoverse-repo PATH."
        )
        raise RuntimeError(f"Unable to import official EgoVerse APIs. {hint}") from exc

    return EgoVerseApi(
        create_default_engine=create_default_engine,
        episode_table_to_df=episode_table_to_df,
        DatasetFilter=DatasetFilter,
        S3EpisodeResolver=S3EpisodeResolver,
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _records_from_dataframe(dataframe: Any) -> list[dict[str, Any]]:
    if not hasattr(dataframe, "to_dict"):
        raise RuntimeError("episode_table_to_df() did not return a DataFrame-like object")
    return [dict(row) for row in dataframe.to_dict(orient="records")]


def query_available_episodes(api: EgoVerseApi) -> list[dict[str, Any]]:
    """Query current SQL metadata and retain downloadable, non-deleted rows."""
    engine = api.create_default_engine()
    try:
        dataframe = api.episode_table_to_df(engine)
    finally:
        dispose = getattr(engine, "dispose", None)
        if callable(dispose):
            dispose()

    active_filter = api.DatasetFilter()
    rows = _records_from_dataframe(dataframe)
    return [
        row
        for row in rows
        if active_filter.matches(row)
        and not _is_missing(row.get("episode_hash"))
        and not _is_missing(row.get("zarr_processed_path"))
    ]


def task_candidates(
    rows: Sequence[dict[str, Any]], task: str, dataset_filter_type: DatasetFilterType
) -> list[EpisodeChoice]:
    if task.casefold() in ALL_TASK_SENTINELS:
        task_filter = dataset_filter_type()
    else:
        # repr() safely quotes a user-provided task inside the official filter expression.
        task_filter = dataset_filter_type(
            filter_lambdas=[f"lambda row: row.get('task') == {task!r}"]
        )

    choices = []
    for row in rows:
        if not task_filter.matches(row):
            continue
        choices.append(
            EpisodeChoice(
                episode_id=_text(row.get("episode_hash")),
                task=_text(row.get("task")),
                lab=_text(row.get("lab")) or "unknown",
                scene=_text(row.get("scene")) or "unknown",
                demonstrator=(
                    _text(row.get("operator"))
                    or _text(row.get("demonstrator"))
                    or "unknown"
                ),
                zarr_processed_path=_text(row.get("zarr_processed_path")),
            )
        )
    return sorted(choices, key=lambda item: item.episode_id)


def select_episodes(
    candidates: Sequence[EpisodeChoice], max_episodes: int, seed: int
) -> list[EpisodeChoice]:
    """Return a stable seeded sample, independent of SQL row ordering."""
    ordered = sorted(candidates, key=lambda item: item.episode_id)
    count = len(ordered) if max_episodes == 0 else min(max_episodes, len(ordered))
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda item: item.episode_id)


def print_selection(
    selected: Sequence[EpisodeChoice], *, available_count: int, seed: int
) -> None:
    print(f"Available matching episodes: {available_count}")
    print(f"Selected episodes: {len(selected)} (seed={seed})")
    if not selected:
        return

    headers = ("episode_id", "lab", "scene", "demonstrator")
    rows = [
        (choice.episode_id, choice.lab, choice.scene, choice.demonstrator)
        for choice in selected
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _selected_hash_filter(api: EgoVerseApi, selected_ids: Sequence[str]) -> Any:
    exact_ids = tuple(selected_ids)
    return api.DatasetFilter(
        filter_lambdas=[
            f"lambda row: row.get('episode_hash') in {exact_ids!r}"
        ]
    )


def download_selection(
    api: EgoVerseApi,
    selected: Sequence[EpisodeChoice],
    *,
    output_dir: Path,
    bucket: str,
    workers: int,
) -> None:
    selected_ids = [choice.episode_id for choice in selected]
    filters = _selected_hash_filter(api, selected_ids)
    synced = api.S3EpisodeResolver.sync_from_filters(
        bucket_name=bucket,
        filters=filters,
        local_dir=output_dir,
        numworkers=workers,
    )
    synced_ids = {str(episode_id) for _, episode_id in synced}
    if synced_ids != set(selected_ids):
        raise RuntimeError(
            "Official resolver result did not match the exact selected episode IDs: "
            f"expected {len(selected_ids)}, resolved {len(synced_ids)}"
        )


def write_manifest(
    output_dir: Path,
    selected: Sequence[EpisodeChoice],
    *,
    seed: int,
    requested_task: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "selection_manifest.json"
    payload = {
        "requested_task": requested_task,
        "seed": seed,
        "episode_count": len(selected),
        "episodes": [asdict(choice) for choice in selected],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace, api: EgoVerseApi) -> int:
    print("Querying the official EgoVerse episode table before download...")
    available = query_available_episodes(api)
    candidates = task_candidates(available, args.task, api.DatasetFilter)
    if not candidates:
        tasks = sorted({_text(row.get("task")) for row in available if row.get("task")})
        preview = ", ".join(tasks[:20])
        raise RuntimeError(
            f"No downloadable episodes matched exact task {args.task!r}. "
            f"Available task examples: {preview or 'none'}"
        )

    selected = select_episodes(candidates, args.max_episodes, args.seed)
    if len(selected) == len(available) and not args.confirm_complete_dataset:
        raise RuntimeError(
            "This selection resolves to the complete downloadable dataset. "
            "Re-run with --confirm-complete-dataset only if that is intentional."
        )

    print_selection(selected, available_count=len(candidates), seed=args.seed)
    if args.dry_run:
        print("Dry run: no files were downloaded or written.")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    print(f"Downloading only the {len(selected)} selected Zarr episodes to {output_dir}")
    download_selection(
        api,
        selected,
        output_dir=output_dir,
        bucket=args.bucket,
        workers=args.workers,
    )
    manifest_path = write_manifest(
        output_dir,
        selected,
        seed=args.seed,
        requested_task=args.task,
    )
    print(f"Download complete. Manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None, api: EgoVerseApi | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_safety_args(args, parser)
    official_api = api if api is not None else load_official_api(args.egoverse_repo)
    try:
        return run(args, official_api)
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
