import unittest

from modal_attempt_clustering import ClusteringConfig, cluster_task


def pair(first, second, score):
    return {
        "attempt_a": first,
        "attempt_b": second,
        "trajectory_similarity": score,
        "orientation_similarity": score,
        "coordination_similarity": score,
        "dynamics_similarity": score,
        "overall_similarity": score,
    }


class ClusteringTests(unittest.TestCase):
    def test_clusters_neighbors_and_medoids(self):
        payload = {
            "task_id": "folding clothes",
            "attempt_ids": ["e:0", "e:1", "e:2", "e:3", "e:4"],
            "pairwise": [
                pair("e:0", "e:1", 0.96),
                pair("e:0", "e:2", 0.94),
                pair("e:1", "e:2", 0.95),
                pair("e:0", "e:3", 0.40),
                pair("e:0", "e:4", 0.42),
                pair("e:1", "e:3", 0.41),
                pair("e:1", "e:4", 0.43),
                pair("e:2", "e:3", 0.39),
                pair("e:2", "e:4", 0.40),
                pair("e:3", "e:4", 0.93),
            ],
        }
        result = cluster_task(payload, ClusteringConfig(0.90, 10))
        self.assertEqual(result["metrics"]["clusters"], 2)
        self.assertEqual(result["metrics"]["attempts_kept"], 2)
        self.assertEqual(result["metrics"]["attempts_dropped"], 3)
        attempts = {item["attempt_id"]: item for item in result["attempts"]}
        self.assertEqual(attempts["e:1"]["decision"], "KEEP")
        self.assertEqual(attempts["e:0"]["represented_by"], "e:1")
        self.assertEqual(attempts["e:0"]["similar_attempts"][0]["attempt_id"], "e:1")
        self.assertEqual(attempts["e:0"]["similar_attempts"][0]["similarity"]["overall"], 0.96)

    def test_average_linkage_drop_guard_promotes_weak_medoid_match(self):
        # Average linkage can merge all three at a 0.80 threshold even though
        # a↔c falls below it. The post-cluster guard must preserve c.
        payload = {
            "task_id": "task",
            "attempt_ids": ["a:0", "b:0", "c:0"],
            "pairwise": [
                pair("a:0", "b:0", 0.90),
                pair("a:0", "c:0", 0.75),
                pair("b:0", "c:0", 0.86),
            ],
        }
        result = cluster_task(payload, ClusteringConfig(0.80, 10))
        decisions = {item["attempt_id"]: item for item in result["attempts"]}
        dropped = [item for item in decisions.values() if item["decision"] == "DROP"]
        self.assertTrue(all(item["similarity_to_representative"] >= 0.80 for item in dropped))

    def test_singleton_is_kept(self):
        payload = {"task_id": "task", "attempt_ids": ["a:0"], "pairwise": []}
        result = cluster_task(payload, ClusteringConfig())
        self.assertEqual(result["attempts"][0]["decision"], "KEEP")
        self.assertEqual(result["metrics"]["clusters"], 1)


if __name__ == "__main__":
    unittest.main()
