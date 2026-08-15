import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from modal_attempt_features import (
    SimilarityConfig,
    _resample_positions,
    _resample_rotations,
    build_attempt_features,
    compare_attempts,
    coverage_metrics,
    greedy_curate,
)


def pose_stream(xyz, rotations=None):
    xyz = np.asarray(xyz, dtype=np.float64)
    rotations = rotations or Rotation.identity(len(xyz))
    xyzw = rotations.as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return np.column_stack((xyz, wxyz))


def synthetic_feature(episode="episode-a", attempt_id=0, offset=0.0):
    timestamps = np.linspace(0.0, 2.0, 21)
    progress = timestamps / timestamps[-1]
    left = np.column_stack((0.3 * progress + offset, 0.1 * progress, np.zeros_like(progress)))
    right = np.column_stack((0.3 * progress + offset, -0.1 * progress, np.zeros_like(progress)))
    head = np.zeros((len(progress), 3))
    rotations = Rotation.from_euler("z", 30.0 * progress, degrees=True)
    attempt = {"attempt_id": attempt_id, "start_idx": 0, "end_idx": len(progress)}
    return build_attempt_features(
        episode_id=episode,
        task_id="fold shirt",
        attempt=attempt,
        left_pose=pose_stream(left, rotations),
        right_pose=pose_stream(right, rotations),
        head_pose=pose_stream(head),
        timestamps=timestamps,
    )


class ResamplingTests(unittest.TestCase):
    def test_position_interpolation_has_requested_shape_and_endpoints(self):
        values = np.asarray([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
        result = _resample_positions(values, np.asarray([2.0, 4.0]), 32)
        self.assertEqual(result.shape, (32, 3))
        np.testing.assert_allclose(result[0], values[0])
        np.testing.assert_allclose(result[-1], values[-1])

    def test_slerp_reaches_rotation_endpoints(self):
        rotations = Rotation.from_euler("z", [0.0, 90.0], degrees=True)
        result = _resample_rotations(rotations, np.asarray([0.0, 1.0]), 32)
        self.assertEqual(len(result), 32)
        np.testing.assert_allclose(result[0].as_matrix(), rotations[0].as_matrix(), atol=1e-8)
        np.testing.assert_allclose(result[-1].as_matrix(), rotations[-1].as_matrix(), atol=1e-8)


class FeatureTests(unittest.TestCase):
    def test_feature_schema_and_head_translation_invariance(self):
        first = synthetic_feature()
        self.assertEqual(np.asarray(first["trajectory"]["left_xyz"]).shape, (32, 3))
        self.assertEqual(np.asarray(first["orientation"]["left"]).shape, (32, 6))
        self.assertEqual(np.asarray(first["coordination"]["relative_hand_position"]).shape, (32, 3))
        self.assertGreater(first["dynamics"]["left"]["path_length"], 0.0)

    def test_identical_attempts_have_unit_similarity(self):
        first = synthetic_feature()
        second = copy.deepcopy(first)
        second["attempt_id"] = "episode-b:0"
        result = compare_attempts(first, second, SimilarityConfig())
        self.assertAlmostEqual(result["overall_similarity"], 1.0)
        self.assertAlmostEqual(result["orientation_similarity"], 1.0)

    def test_physical_trajectory_difference_reduces_similarity(self):
        first = synthetic_feature()
        second = synthetic_feature(episode="episode-b", offset=0.4)
        result = compare_attempts(first, second, SimilarityConfig())
        self.assertLess(result["trajectory_similarity"], 0.1)
        self.assertLess(result["overall_similarity"], 0.8)


class CurationTests(unittest.TestCase):
    def test_greedy_drop_and_coverage(self):
        attempts = [
            {"attempt_id": "a"},
            {"attempt_id": "b"},
            {"attempt_id": "c"},
        ]
        pairs = [
            {"attempt_a": "a", "attempt_b": "b", "overall_similarity": 0.95},
            {"attempt_a": "a", "attempt_b": "c", "overall_similarity": 0.50},
            {"attempt_a": "b", "attempt_b": "c", "overall_similarity": 0.55},
        ]
        decisions = greedy_curate(attempts, pairs, 0.90)
        self.assertEqual([item["decision"] for item in decisions], ["KEEP", "DROP", "KEEP"])
        self.assertEqual(decisions[1]["represented_by"], "a")
        metrics = coverage_metrics(decisions)
        self.assertEqual(metrics["attempts_kept"], 2)
        self.assertAlmostEqual(metrics["mean_behavioral_coverage"], (1.0 + 0.95 + 1.0) / 3.0)


if __name__ == "__main__":
    unittest.main()
