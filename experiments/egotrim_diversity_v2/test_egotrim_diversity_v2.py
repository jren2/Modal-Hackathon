"""Deterministic synthetic validation for the isolated EgoTrim V2 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modal_app import _egoverse_clip_rows
from run_egotrim_diversity_v2 import (
    FINGERTIP_DIMS,
    N_STEPS,
    TRAJECTORY_DIMS,
    BalancedEmbeddingModel,
    PipelineConfig,
    PoseSegment,
    TrackingError,
    _facility_similarity,
    _select_from_order,
    cluster_within_verbs,
    composition_metrics,
    distinctiveness_scores,
    extract_raw_feature,
    facility_value,
    load_pose_segment,
    pairwise_distances,
    run_pipeline_from_features,
    select_facility_subset,
    vendi_from_kernel,
)


def test_egoverse_clip_manifest_adapter_preserves_real_paths_and_timing(
    tmp_path: Path,
) -> None:
    clips_root = tmp_path / "segments"
    episodes_root = tmp_path / "episodes"
    camera_root = clips_root / "episode_001" / "front_1"
    episode_root = episodes_root / "episode_001"
    camera_root.mkdir(parents=True)
    episode_root.mkdir(parents=True)
    (camera_root / "000000.mp4").write_bytes(b"model-clip")
    (camera_root / "manifest.json").write_text(
        json.dumps(
            {
                "episode": "episode_001",
                "fps": 30,
                "segments": [
                    {
                        "file": "000000.mp4",
                        "start_frame": 45,
                        "end_frame_exclusive": 75,
                        "start_seconds": 1.5,
                        "duration_seconds": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (episode_root / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "task_name": "fold_clothes",
                    "fps": 30,
                    "total_frames": 90,
                },
            }
        ),
        encoding="utf-8",
    )

    rows, report = _egoverse_clip_rows(clips_root, episodes_root)

    assert len(rows) == 1
    assert rows[0]["segment_id"] == "episode_001:front_1:000000"
    assert rows[0]["canonical_verb"] == "fold_clothes"
    assert rows[0]["start_time"] == 1.5
    assert rows[0]["end_time"] == 2.5
    assert rows[0]["start_frame"] == 45
    assert rows[0]["end_frame_exclusive"] == 75
    assert rows[0]["tracking_valid_frac"] == 1.0
    assert Path(rows[0]["video_path"]).resolve() == (camera_root / "000000.mp4").resolve()
    assert Path(rows[0]["source_data_path"]).resolve() == episode_root.resolve()
    assert report["adapter"] == "egoverse_fixed_window_clip_manifests_v1"
    assert "not semantic action-cycle" in report["input_semantics"]


def test_pose_loader_prefers_exact_frame_bounds_over_gapped_timestamps(
    tmp_path: Path,
) -> None:
    pose_path = tmp_path / "episode_pose.npz"
    frames = 6
    joints = np.full((frames, 2, 21, 3), 1.0, dtype=float)
    timestamps = np.asarray([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    np.savez(
        pose_path,
        hand_joints=joints,
        timestamps=timestamps,
        coordinate_frame=np.asarray("head_frame"),
    )
    row = {
        "episode_id": "episode_pose",
        "segment_id": "clip_001",
        "start_time": 0.2,
        "end_time": 1.2,
        "start_frame": 2,
        "end_frame_exclusive": 5,
        "pose_ref": str(pose_path),
    }

    pose = load_pose_segment(row, tmp_path)

    np.testing.assert_allclose(pose.timestamps, [0.2, 0.2 + 1.0 / 3.0, 0.2 + 2.0 / 3.0])
    assert pose.joints.shape == (3, 2, 21, 3)
    assert pose.schema_evidence["frame_mapping"] == "exact_manifest_frame_bounds:2:5"
    assert (
        pose.schema_evidence["timestamp_mapping"]
        == "nominal_clip_timeline_from_exact_manifest_frame_bounds"
    )
    assert pose.schema_evidence["source_timestamp_delta_seconds"]["maximum"] == pytest.approx(9.8)


def _synthetic_pose(frames: int = 12, motion_scale: float = 1.0) -> PoseSegment:
    """Return nonzero head-frame joints with deliberately different hand motion."""
    timestamps = np.linspace(0.0, 1.0, frames)
    phase = timestamps / timestamps[-1]
    wrists = np.empty((frames, 2, 3), dtype=float)
    wrists[:, 0, :] = np.column_stack(
        [1.0 + motion_scale * phase, 1.5 + 0.1 * phase**2, 2.0 + 0.05 * phase]
    )
    wrists[:, 1, :] = np.column_stack(
        [2.0 - 0.1 * phase, 2.5 + 2.0 * motion_scale * phase, 3.0 + 0.08 * phase**2]
    )
    joints = np.repeat(wrists[:, :, None, :], 21, axis=2)
    for joint in range(1, 21):
        joints[:, 0, joint, :] += np.array([0.005 * joint, 0.01, 0.015])
        joints[:, 1, joint, :] += np.array([-0.007 * joint, 0.02, 0.025])
    return PoseSegment(
        timestamps=timestamps,
        joints=joints,
        wrists=None,
        coordinate_mode="synthetic_declared_head_frame",
        source_path="synthetic_pose",
        schema_evidence={"fixture": True},
    )


def _selection_metadata(embedding: np.ndarray, verbs: list[str] | None = None) -> pd.DataFrame:
    count = len(embedding)
    if verbs is None:
        verbs = ["fold"] * count
    return pd.DataFrame(
        {
            "episode_id": [f"ep_{i // 3:02d}" for i in range(count)],
            "segment_id": [f"seg_{i:03d}" for i in range(count)],
            "canonical_verb": verbs,
            "start_time": np.arange(count, dtype=float),
            "end_time": np.arange(count, dtype=float) + 1.0,
            "duration": np.ones(count, dtype=float),
            "tracking_quality": np.full(count, 0.99),
            "potential_tracking_outlier": np.zeros(count, dtype=bool),
        }
    )


def test_resampling_dimensions_and_left_right_hand_separation() -> None:
    feature = extract_raw_feature(_synthetic_pose(frames=9), duration=1.0)

    assert feature.trajectory.shape == (TRAJECTORY_DIMS,)
    assert feature.fingertips.shape == (FINGERTIP_DIMS,)
    assert TRAJECTORY_DIMS == N_STEPS * 3 * 2
    assert FINGERTIP_DIMS == N_STEPS * 5 * 3 * 2

    trajectory = feature.trajectory.reshape(N_STEPS, 3, 2)
    fingertips = feature.fingertips.reshape(N_STEPS, 5, 3, 2)
    np.testing.assert_allclose(trajectory[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(trajectory[-1, 0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(trajectory[-1, 1, 1], 2.0, atol=1e-12)
    assert not np.allclose(trajectory[:, :, 0], trajectory[:, :, 1])
    np.testing.assert_allclose(fingertips[:, 0, 0, 0], 0.020, atol=1e-12)
    np.testing.assert_allclose(fingertips[:, 0, 0, 1], -0.028, atol=1e-12)


def test_duplicate_motion_has_zero_distance_and_different_motion_is_more_novel() -> None:
    pytest.importorskip("sklearn")
    scales = [1.0, 1.0, 0.75, 0.85, 1.15, 1.30, 1.60, 2.50]
    features = [extract_raw_feature(_synthetic_pose(motion_scale=s), 0.8 + 0.05 * i) for i, s in enumerate(scales)]
    # The first two examples must be exact duplicates, including duration.
    features[1] = extract_raw_feature(_synthetic_pose(motion_scale=1.0), 0.8)
    model = BalancedEmbeddingModel(max_components=5, variance_target=1.0)
    embeddings = model.fit_transform(
        np.stack([f.trajectory for f in features]),
        np.stack([f.fingertips for f in features]),
        np.asarray([f.log_duration for f in features]),
    )["combined"]

    distances = pairwise_distances(embeddings)
    assert distances[0, 1] < 1e-8
    assert distances[0, -1] > distances[0, 1] + 1.0

    metadata = _selection_metadata(embeddings)
    scores = distinctiveness_scores(
        embeddings, metadata, np.asarray(["fold__0"] * len(metadata)), k=1
    )
    assert scores.loc[0, "mean_same_verb_knn_distance"] < 1e-8
    assert scores.loc[len(scores) - 1, "distinctiveness_percentile"] > scores.loc[0, "distinctiveness_percentile"]


def test_balanced_blocks_are_unit_scaled_and_invariant_to_tip_magnitude() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(20260815)
    count = 40
    trajectory = rng.normal(size=(count, TRAJECTORY_DIMS))
    fingertips = rng.normal(size=(count, FINGERTIP_DIMS))
    duration = rng.normal(size=count)

    ordinary = BalancedEmbeddingModel(max_components=5, variance_target=1.0)
    scaled = BalancedEmbeddingModel(max_components=5, variance_target=1.0)
    embedded = ordinary.fit_transform(trajectory, fingertips, duration)["combined"]
    embedded_large_tips = scaled.fit_transform(
        trajectory, fingertips * 1_000_000.0, duration
    )["combined"]

    np.testing.assert_allclose(embedded.var(axis=0), 1.0, rtol=1e-7, atol=1e-7)
    trajectory_energy = np.mean(
        np.sum(embedded[:, : ordinary.trajectory_components_] ** 2, axis=1)
    )
    fingertip_start = ordinary.trajectory_components_
    fingertip_end = fingertip_start + ordinary.fingertip_components_
    fingertip_energy = np.mean(np.sum(embedded[:, fingertip_start:fingertip_end] ** 2, axis=1))
    expected_ratio = ordinary.trajectory_components_ / ordinary.fingertip_components_
    assert trajectory_energy / fingertip_energy == pytest.approx(expected_ratio, rel=1e-6)
    np.testing.assert_allclose(
        pairwise_distances(embedded),
        pairwise_distances(embedded_large_tips),
        rtol=1e-6,
        atol=1e-6,
    )


def test_short_internal_tracking_gap_is_interpolated_without_zero_fill() -> None:
    pose = _synthetic_pose(frames=10)
    pose.joints[4, 0, 8, :] = np.nan
    feature = extract_raw_feature(pose, duration=1.0, max_interp_gap=1)

    assert feature.interpolated_value_count == 3
    assert feature.missing_value_rate > 0.0
    assert feature.measured_tracking_valid_frac == pytest.approx(0.9)
    assert np.isfinite(feature.trajectory).all()
    assert np.isfinite(feature.fingertips).all()


@pytest.mark.parametrize("missing_slice", [slice(0, 1), slice(4, 6)])
def test_edge_or_long_tracking_gaps_are_rejected(missing_slice: slice) -> None:
    pose = _synthetic_pose(frames=10)
    pose.joints[missing_slice, 0, 8, :] = np.nan
    with pytest.raises(TrackingError, match="remain missing"):
        extract_raw_feature(pose, duration=1.0, max_interp_gap=1)


def test_substantial_missing_tracking_is_rejected_before_interpolation() -> None:
    pose = _synthetic_pose(frames=10)
    pose.joints[2:6, 0, 8, :] = np.nan
    with pytest.raises(TrackingError, match="tracking fraction"):
        extract_raw_feature(pose, duration=1.0, max_interp_gap=10)


def test_same_seed_produces_identical_clusters_and_selection() -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(42)
    centers = np.asarray([[-4.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
    embedding = np.vstack(
        [center + rng.normal(0.0, 0.08, size=(6, 2)) for center in centers]
    )
    verbs = ["fold"] * 6 + ["smooth"] * 6 + ["adjust"] * 6
    metadata = _selection_metadata(embedding, verbs)

    labels_a, summary_a, audit_a = cluster_within_verbs(embedding, metadata, seed=73)
    labels_b, summary_b, audit_b = cluster_within_verbs(embedding, metadata, seed=73)
    np.testing.assert_array_equal(labels_a, labels_b)
    assert summary_a == summary_b
    assert audit_a == audit_b

    metadata["potential_tracking_outlier"] = False
    selected_a, reasons_a, selection_audit_a = select_facility_subset(
        embedding, metadata, count_budget=6, duration_budget=6.0
    )
    selected_b, reasons_b, selection_audit_b = select_facility_subset(
        embedding, metadata, count_budget=6, duration_budget=6.0
    )
    assert selected_a == selected_b
    assert reasons_a == reasons_b
    assert selection_audit_a == selection_audit_b


def test_facility_curation_beats_seeded_random_on_clustered_data() -> None:
    offsets = np.linspace(-0.08, 0.08, 5)
    embedding = np.vstack(
        [
            np.column_stack([np.full(5, center) + offsets, np.zeros(5)])
            for center in (-8.0, 0.0, 8.0)
        ]
    )
    metadata = _selection_metadata(embedding)
    curated, _, audit = select_facility_subset(
        embedding, metadata, count_budget=3, duration_budget=3.0
    )
    random_order = np.random.default_rng(2).permutation(len(metadata))
    random_selected = _select_from_order(
        random_order, metadata["duration"].to_numpy(), count_budget=3, duration_budget=3.0
    )
    similarity, _ = _facility_similarity(embedding)

    assert len(curated) == len(random_selected) == 3
    assert audit["selected_duration"] <= 3.0
    assert facility_value(similarity, curated) >= facility_value(similarity, random_selected)


def test_vendi_and_composition_invariants() -> None:
    assert vendi_from_kernel(np.ones((5, 5))) == pytest.approx(1.0)
    assert vendi_from_kernel(np.eye(5)) == pytest.approx(5.0)
    composition = composition_metrics(["fold", "fold", "smooth", "smooth"])
    assert composition["verb_coverage"] == 2
    assert composition["shannon_effective_number_of_verbs"] == pytest.approx(2.0)
    assert "not Vendi" in composition["definition_note"]


def _full_metadata(count: int) -> pd.DataFrame:
    verbs = np.resize(np.asarray(["fold", "smooth", "adjust"]), count)
    durations = 0.8 + 0.02 * np.arange(count)
    return pd.DataFrame(
        {
            "episode_id": [f"episode_{i // 3:02d}" for i in range(count)],
            "segment_id": [f"segment_{i:03d}" for i in range(count)],
            "canonical_verb": verbs,
            "start_time": np.arange(count, dtype=float),
            "end_time": np.arange(count, dtype=float) + durations,
            "duration": durations,
            "tracking_valid_frac": np.full(count, 0.98),
            "measured_tracking_valid_frac": np.full(count, 1.0),
            "tracking_quality": np.full(count, 0.98),
            "missing_value_rate": np.zeros(count),
            "interpolated_value_count": np.zeros(count, dtype=int),
            "coordinate_mode": ["synthetic_declared_head_frame"] * count,
            "pose_source_path": [f"pose_{i:03d}.npz" for i in range(count)],
            "video_path": [f"video_{i // 3:02d}.mp4" for i in range(count)],
            "source_data_path": [f"pose_{i:03d}.npz" for i in range(count)],
        }
    )


def test_small_pipeline_writes_strict_output_contract(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(991)
    count = 12
    metadata = _full_metadata(count)
    trajectory = rng.normal(size=(count, TRAJECTORY_DIMS))
    fingertips = rng.normal(size=(count, FINGERTIP_DIMS))
    log_duration = np.log(metadata["duration"].to_numpy())
    output_dir = tmp_path / "isolated_output"
    config = PipelineConfig(
        segments=tmp_path / "unused_segments.csv",
        pose_root=tmp_path / "unused_poses",
        output_dir=output_dir,
        budget_frac=0.5,
        seed=42,
        pca_components=3,
        pca_variance_target=1.0,
        baseline_runs=2,
        enforce_local_isolation=False,
    )

    metrics = run_pipeline_from_features(
        metadata,
        trajectory,
        fingertips,
        log_duration,
        rejected_rows=[],
        schema_report={"status": "synthetic_test"},
        config=config,
    )

    required = {
        "segment_scores.csv",
        "episode_scores.csv",
        "cluster_summary.csv",
        "nearest_neighbors.csv",
        "subset_manifest.csv",
        "subset_manifest.json",
        "metrics.json",
        "dashboard_data.json",
        "schema_report.json",
    }
    assert required <= {path.name for path in output_dir.iterdir() if path.is_file()}
    assert (output_dir / "models").is_dir()
    assert (output_dir / "charts").is_dir()
    assert len(list((output_dir / "charts").glob("*.png"))) == 9
    assert (output_dir / "models" / "trajectory_scaler.joblib").is_file()
    assert (output_dir / "models" / "fingertip_pca.joblib").is_file()
    assert (output_dir / "models" / "final_combined_scaler.joblib").is_file()

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    for filename in ("metrics.json", "dashboard_data.json", "subset_manifest.json"):
        json.loads(
            (output_dir / filename).read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_constant,
        )

    manifest_csv = pd.read_csv(output_dir / "subset_manifest.csv")
    manifest_json = json.loads((output_dir / "subset_manifest.json").read_text(encoding="utf-8"))
    required_manifest_columns = {
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
    }
    assert required_manifest_columns <= set(manifest_csv.columns)
    assert manifest_csv["segment_id"].tolist() == [row["segment_id"] for row in manifest_json]
    assert metrics["feature_balance"]["blocks_standardized_independently"] is True
    assert metrics["feature_balance"]["final_embedding_standardized"] is True
    assert "no downstream robot-policy improvement" in metrics["claims_guardrail"]
