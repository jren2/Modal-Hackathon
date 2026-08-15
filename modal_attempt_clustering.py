"""Cluster similar EgoVerse attempts and emit dashboard-friendly neighbor data.

This is intentionally separate from feature extraction and consumes its
task-level pairwise similarity outputs without modifying them.

Run against every task currently available in the Modal Volume:
    modal run modal_attempt_clustering.py

Outputs:
    /egoverse/attempt_clusters/summary.json
    /egoverse/attempt_clusters/attempt_index.json
    /egoverse/attempt_clusters/tasks/<task-key>.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import modal


VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
ATTEMPTS = MOUNT / "attempts"
SIMILARITY = MOUNT / "attempt_similarity"
OUTPUT = MOUNT / "attempt_clusters"

app = modal.App("egoverse-attempt-clustering")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.2.6",
    "scipy==1.15.3",
)


@dataclass(frozen=True)
class ClusteringConfig:
    similarity_threshold: float = 0.90
    neighbor_limit: int = 50

    def validate(self) -> None:
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if self.neighbor_limit < 1:
            raise ValueError("neighbor_limit must be at least 1")


def _pair_lookup(pairwise: list[dict]) -> dict[frozenset[str], dict]:
    lookup = {}
    for pair in pairwise:
        key = frozenset((pair["attempt_a"], pair["attempt_b"]))
        if len(key) != 2:
            raise ValueError("Pairwise records must contain two distinct attempts")
        if key in lookup:
            raise ValueError(f"Duplicate pairwise record: {sorted(key)}")
        lookup[key] = pair
    return lookup


def _similarity_matrix(attempt_ids: list[str], pair_lookup: dict):
    import numpy as np

    size = len(attempt_ids)
    matrix = np.eye(size, dtype=np.float64)
    for first in range(size):
        for second in range(first + 1, size):
            key = frozenset((attempt_ids[first], attempt_ids[second]))
            if key not in pair_lookup:
                raise ValueError(f"Missing similarity pair: {attempt_ids[first]} ↔ {attempt_ids[second]}")
            score = float(pair_lookup[key]["overall_similarity"])
            if not 0 <= score <= 1:
                raise ValueError(f"Similarity outside [0, 1]: {score}")
            matrix[first, second] = matrix[second, first] = score
    return matrix


def _average_linkage_assignments(similarity_matrix, threshold: float):
    """Return zero-based cluster labels using average-linkage dissimilarity."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    size = len(similarity_matrix)
    if size == 0:
        return np.asarray([], dtype=np.int64)
    if size == 1:
        return np.asarray([0], dtype=np.int64)
    distance = np.clip(1.0 - similarity_matrix, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    hierarchy = linkage(squareform(distance, checks=True), method="average")
    raw = fcluster(hierarchy, t=1.0 - threshold, criterion="distance")
    return raw.astype(np.int64) - 1


def _canonical_clusters(attempt_ids: list[str], raw_labels) -> list[list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(raw_labels):
        grouped.setdefault(int(label), []).append(index)
    clusters = list(grouped.values())
    for members in clusters:
        members.sort(key=lambda index: attempt_ids[index])
    clusters.sort(key=lambda members: attempt_ids[members[0]])
    return clusters


def _medoid(members: list[int], attempt_ids: list[str], matrix, confidences: dict[str, float]) -> int:
    import numpy as np

    if len(members) == 1:
        return members[0]
    candidates = []
    for member in members:
        peers = [peer for peer in members if peer != member]
        mean_similarity = float(np.mean([matrix[member, peer] for peer in peers]))
        candidates.append(
            (
                mean_similarity,
                float(confidences.get(attempt_ids[member], 0.0)),
                attempt_ids[member],
                member,
            )
        )
    # Centrality is primary, extraction confidence breaks real ties, then ID.
    return sorted(candidates, key=lambda value: (-value[0], -value[1], value[2]))[0][3]


def _curate_cluster(
    members: list[int],
    medoid: int,
    attempt_ids: list[str],
    matrix,
    threshold: float,
) -> tuple[list[int], dict[int, dict]]:
    """Keep a medoid and promote any member lacking a threshold-safe representative."""
    kept = [medoid]
    decisions = {
        medoid: {
            "decision": "KEEP",
            "reason": "cluster_medoid",
            "represented_by": attempt_ids[medoid],
            "similarity_to_representative": 1.0,
        }
    }
    remaining = sorted(
        (member for member in members if member != medoid),
        key=lambda member: (-matrix[member, medoid], attempt_ids[member]),
    )
    for member in remaining:
        representative = max(
            kept,
            key=lambda candidate: (matrix[member, candidate], attempt_ids[candidate]),
        )
        score = float(matrix[member, representative])
        if score >= threshold:
            decisions[member] = {
                "decision": "DROP",
                "reason": "represented_above_threshold",
                "represented_by": attempt_ids[representative],
                "similarity_to_representative": score,
            }
        else:
            kept.append(member)
            decisions[member] = {
                "decision": "KEEP",
                "reason": "representative_threshold_guard",
                "represented_by": attempt_ids[member],
                "similarity_to_representative": 1.0,
            }
    return kept, decisions


def _component_view(pair: dict) -> dict:
    return {
        "overall": float(pair["overall_similarity"]),
        "trajectory": float(pair["trajectory_similarity"]),
        "orientation": float(pair["orientation_similarity"]),
        "coordination": float(pair["coordination_similarity"]),
        "dynamics": float(pair["dynamics_similarity"]),
        "distances": pair.get("distances", {}),
    }


def _metrics(attempt_ids: list[str], decisions: dict[int, dict], matrix) -> dict:
    import numpy as np

    if not attempt_ids:
        return {
            "original_attempts": 0,
            "clusters": 0,
            "attempts_kept": 0,
            "attempts_dropped": 0,
            "retained_fraction": 0.0,
            "reduction_fraction": 0.0,
            "mean_dropped_representation_similarity": None,
            "minimum_dropped_representation_similarity": None,
            "mean_nearest_kept_coverage": 0.0,
            "worst_case_nearest_kept_coverage": 0.0,
        }
    kept = [index for index, decision in decisions.items() if decision["decision"] == "KEEP"]
    dropped_scores = [
        decision["similarity_to_representative"]
        for decision in decisions.values()
        if decision["decision"] == "DROP"
    ]
    coverage = [float(max(matrix[index, candidate] for candidate in kept)) for index in range(len(attempt_ids))]
    return {
        "original_attempts": len(attempt_ids),
        "attempts_kept": len(kept),
        "attempts_dropped": len(attempt_ids) - len(kept),
        "retained_fraction": len(kept) / len(attempt_ids),
        "reduction_fraction": 1.0 - len(kept) / len(attempt_ids),
        "mean_dropped_representation_similarity": float(np.mean(dropped_scores)) if dropped_scores else None,
        "minimum_dropped_representation_similarity": float(np.min(dropped_scores)) if dropped_scores else None,
        "mean_nearest_kept_coverage": float(np.mean(coverage)),
        "worst_case_nearest_kept_coverage": float(np.min(coverage)),
    }


def _aggregate_metrics(rows: list[dict]) -> dict:
    original = sum(row["original_attempts"] for row in rows)
    kept = sum(row["attempts_kept"] for row in rows)
    dropped = sum(row["attempts_dropped"] for row in rows)
    dropped_rows = [row for row in rows if row["attempts_dropped"]]
    return {
        "original_attempts": original,
        "clusters": sum(row.get("clusters", 0) for row in rows),
        "attempts_kept": kept,
        "attempts_dropped": dropped,
        "retained_fraction": kept / original if original else 0.0,
        "reduction_fraction": dropped / original if original else 0.0,
        "mean_dropped_representation_similarity": (
            sum(
                row["mean_dropped_representation_similarity"] * row["attempts_dropped"]
                for row in dropped_rows
            )
            / dropped
            if dropped
            else None
        ),
        "minimum_dropped_representation_similarity": (
            min(row["minimum_dropped_representation_similarity"] for row in dropped_rows)
            if dropped_rows
            else None
        ),
        "mean_nearest_kept_coverage": (
            sum(row["mean_nearest_kept_coverage"] * row["original_attempts"] for row in rows)
            / original
            if original
            else 0.0
        ),
        "worst_case_nearest_kept_coverage": (
            min(row["worst_case_nearest_kept_coverage"] for row in rows if row["original_attempts"])
            if original
            else 0.0
        ),
    }


def cluster_task(
    task_payload: dict,
    config: ClusteringConfig,
    *,
    confidences: dict[str, float] | None = None,
    include_neighbors: bool = True,
) -> dict:
    """Cluster one task and produce records suitable for an attempt detail UI."""
    config.validate()
    confidences = confidences or {}
    attempt_ids = list(task_payload.get("attempt_ids", []))
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("attempt_ids must be unique")
    pairs = list(task_payload.get("pairwise", []))
    pair_lookup = _pair_lookup(pairs)
    matrix = _similarity_matrix(attempt_ids, pair_lookup)
    raw_labels = _average_linkage_assignments(matrix, config.similarity_threshold)
    clusters = _canonical_clusters(attempt_ids, raw_labels)

    decisions: dict[int, dict] = {}
    cluster_records = []
    cluster_by_index = {}
    medoid_by_cluster = {}
    for cluster_number, members in enumerate(clusters, start=1):
        medoid = _medoid(members, attempt_ids, matrix, confidences)
        kept, cluster_decisions = _curate_cluster(
            members, medoid, attempt_ids, matrix, config.similarity_threshold
        )
        decisions.update(cluster_decisions)
        for member in members:
            cluster_by_index[member] = cluster_number
        medoid_by_cluster[cluster_number] = medoid
        cluster_records.append(
            {
                "cluster_id": cluster_number,
                "size": len(members),
                "medoid_attempt_id": attempt_ids[medoid],
                "member_attempt_ids": [attempt_ids[index] for index in members],
                "kept_attempt_ids": [attempt_ids[index] for index in kept],
            }
        )

    attempt_records = []
    for index, attempt_id in enumerate(attempt_ids):
        cluster_id = cluster_by_index[index]
        record = {
            "attempt_id": attempt_id,
            "cluster_id": cluster_id,
            "cluster_medoid_attempt_id": attempt_ids[medoid_by_cluster[cluster_id]],
            "extraction_confidence": confidences.get(attempt_id),
            **decisions[index],
        }
        if include_neighbors:
            neighbors = []
            for other_index, other_id in enumerate(attempt_ids):
                if other_index == index:
                    continue
                pair = pair_lookup[frozenset((attempt_id, other_id))]
                neighbors.append(
                    {
                        "attempt_id": other_id,
                        "same_cluster": cluster_by_index[other_index] == cluster_id,
                        "cluster_id": cluster_by_index[other_index],
                        "decision": decisions[other_index]["decision"],
                        "represented_by": decisions[other_index]["represented_by"],
                        "similarity": _component_view(pair),
                    }
                )
            neighbors.sort(key=lambda value: (-value["similarity"]["overall"], value["attempt_id"]))
            record["similar_attempts"] = neighbors[: config.neighbor_limit]
        attempt_records.append(record)

    metrics = _metrics(attempt_ids, decisions, matrix)
    metrics["clusters"] = len(clusters)
    ordered_indices = [
        index
        for members in clusters
        for index in sorted(
            members,
            key=lambda member: (-matrix[member, medoid_by_cluster[cluster_by_index[member]]], attempt_ids[member]),
        )
    ]
    return {
        "schema_version": 1,
        "task_id": task_payload.get("task_id"),
        "clustering": {
            "algorithm": "agglomerative",
            "linkage": "average",
            "distance": "1_minus_overall_similarity",
            **asdict(config),
            "drop_safety": "promote_member_if_no_kept_representative_meets_threshold",
        },
        "source_similarity_config": task_payload.get("similarity_config", {}),
        "metrics": metrics,
        "clusters": cluster_records,
        "attempts": attempt_records,
        "visualization": {
            "attempt_order": [attempt_ids[index] for index in ordered_indices],
            "similarity_matrix": matrix.tolist(),
            "matrix_attempt_order": attempt_ids,
        },
    }


def _load_extraction_confidences(attempt_ids: list[str]) -> dict[str, float]:
    by_episode: dict[str, list[str]] = {}
    for attempt_id in attempt_ids:
        episode_id, _, _ = attempt_id.rpartition(":")
        by_episode.setdefault(episode_id, []).append(attempt_id)
    confidences = {}
    for episode_id in by_episode:
        manifest_path = ATTEMPTS / episode_id / "attempts.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for attempt in manifest.get("attempts", []):
            key = f"{episode_id}:{int(attempt['attempt_id'])}"
            if attempt.get("confidence") is not None:
                confidences[key] = float(attempt["confidence"])
    return confidences


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)


@app.function(image=image, volumes={str(MOUNT): volume}, cpu=4, memory=8192, timeout=60 * 60)
def cluster_all_tasks(
    similarity_threshold: float = 0.90,
    neighbor_limit: int = 50,
    experiment_thresholds: list[float] | None = None,
) -> dict:
    volume.reload()
    config = ClusteringConfig(similarity_threshold, neighbor_limit)
    config.validate()
    experiment_thresholds = experiment_thresholds or [0.95, 0.90, 0.85, 0.80]
    if any(not 0 <= threshold <= 1 for threshold in experiment_thresholds):
        raise ValueError("All experiment thresholds must be in [0, 1]")

    task_paths = sorted((SIMILARITY / "tasks").glob("*.json"))
    if not task_paths:
        raise RuntimeError("No /egoverse/attempt_similarity/tasks/*.json files were found")

    task_summaries = []
    attempt_index = {}
    aggregate_experiments = {threshold: [] for threshold in experiment_thresholds}
    for source_path in task_paths:
        source = json.loads(source_path.read_text())
        confidences = _load_extraction_confidences(source.get("attempt_ids", []))
        result = cluster_task(source, config, confidences=confidences)
        result["threshold_experiments"] = []
        for threshold in experiment_thresholds:
            experiment = cluster_task(
                source,
                ClusteringConfig(threshold, neighbor_limit),
                confidences=confidences,
                include_neighbors=False,
            )
            row = {"threshold": threshold, **experiment["metrics"]}
            result["threshold_experiments"].append(row)
            aggregate_experiments[threshold].append(row)

        destination = OUTPUT / "tasks" / source_path.name
        _write_json_atomic(destination, result)
        task_summaries.append(
            {
                "task_id": result["task_id"],
                "task_key": source_path.stem,
                "result_path": str(destination),
                **result["metrics"],
            }
        )
        for attempt in result["attempts"]:
            attempt_index[attempt["attempt_id"]] = {
                "task_id": result["task_id"],
                "task_key": source_path.stem,
                "result_path": str(destination),
                "cluster_id": attempt["cluster_id"],
                "decision": attempt["decision"],
                "represented_by": attempt["represented_by"],
            }

    aggregate = _aggregate_metrics(task_summaries)
    summary = {
        "schema_version": 1,
        "clustering_config": asdict(config),
        "task_count": len(task_summaries),
        **aggregate,
        "threshold_experiments": [
            {
                "threshold": threshold,
                **_aggregate_metrics(rows),
            }
            for threshold, rows in aggregate_experiments.items()
        ],
        "tasks": task_summaries,
    }
    _write_json_atomic(OUTPUT / "attempt_index.json", attempt_index)
    _write_json_atomic(OUTPUT / "summary.json", summary)
    volume.commit()
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}))
    return summary


@app.local_entrypoint()
def main(
    similarity_threshold: float = 0.90,
    neighbor_limit: int = 50,
    experiment_thresholds: str = "0.95,0.90,0.85,0.80",
):
    thresholds = [float(value.strip()) for value in experiment_thresholds.split(",") if value.strip()]
    summary = cluster_all_tasks.remote(similarity_threshold, neighbor_limit, thresholds)
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=2))
