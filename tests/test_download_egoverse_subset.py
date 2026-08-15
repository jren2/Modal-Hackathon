from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "download_egoverse_subset.py"
)
SPEC = importlib.util.spec_from_file_location("download_egoverse_subset", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeDataFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return list(self.rows)


class FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


class FakeDatasetFilter:
    def __init__(self, filter_lambdas=None):
        self.predicates = [eval(expr) for expr in (filter_lambdas or [])]

    def matches(self, row):
        return not row.get("is_deleted", False) and all(
            predicate(dict(row)) for predicate in self.predicates
        )


class FakeResolver:
    calls = []
    rows = []

    @classmethod
    def sync_from_filters(cls, **kwargs):
        cls.calls.append(kwargs)
        matches = [row for row in cls.rows if kwargs["filters"].matches(row)]
        return [(row["zarr_processed_path"], row["episode_hash"]) for row in matches]


def make_rows(count=15):
    return [
        {
            "episode_hash": f"episode-{index:02d}",
            "task": "put_object_in_container",
            "lab": f"lab-{index % 2}",
            "scene": f"scene-{index % 3}",
            "operator": f"person-{index % 4}",
            "zarr_processed_path": f"s3://rldb/processed/{index}/",
            "is_deleted": False,
        }
        for index in range(count)
    ]


def make_api(rows):
    engine = FakeEngine()
    FakeResolver.calls = []
    FakeResolver.rows = list(rows)
    api = module.EgoVerseApi(
        create_default_engine=lambda: engine,
        episode_table_to_df=lambda unused_engine: FakeDataFrame(rows),
        DatasetFilter=FakeDatasetFilter,
        S3EpisodeResolver=FakeResolver,
    )
    return api, engine


class DownloadEgoVerseSubsetTests(unittest.TestCase):
    def test_task_is_required(self):
        with self.assertRaises(SystemExit):
            module.main([])

    def test_default_is_deterministic_and_capped_at_ten(self):
        rows = make_rows()
        api, engine = make_api(rows)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = module.main(
                ["--task", "put_object_in_container", "--dry-run"], api=api
            )

        self.assertEqual(result, 0)
        self.assertIn("Selected episodes: 10 (seed=42)", output.getvalue())
        self.assertIn("Dry run: no files were downloaded or written.", output.getvalue())
        self.assertEqual(FakeResolver.calls, [])
        self.assertTrue(engine.disposed)

    def test_selection_does_not_depend_on_sql_row_order(self):
        rows = make_rows()
        choices = [
            module.EpisodeChoice(
                episode_id=row["episode_hash"],
                task=row["task"],
                lab=row["lab"],
                scene=row["scene"],
                demonstrator=row["operator"],
                zarr_processed_path=row["zarr_processed_path"],
            )
            for row in rows
        ]
        forward = module.select_episodes(choices, max_episodes=5, seed=7)
        reverse = module.select_episodes(list(reversed(choices)), max_episodes=5, seed=7)
        self.assertEqual(forward, reverse)

    def test_unbounded_and_all_task_requests_require_confirmation(self):
        for argv in (
            ["--task", "put_object_in_container", "--max-episodes", "0"],
            ["--task", "*"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                module.main(argv)

    def test_download_passes_only_selected_hashes_to_official_resolver(self):
        rows = make_rows(8)
        api, _ = make_api(rows)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = module.main(
                    [
                        "--task",
                        "put_object_in_container",
                        "--max-episodes",
                        "3",
                        "--seed",
                        "11",
                        "--output-dir",
                        temp_dir,
                    ],
                    api=api,
                )

            self.assertEqual(result, 0)
            self.assertEqual(len(FakeResolver.calls), 1)
            call = FakeResolver.calls[0]
            matched = {
                row["episode_hash"]
                for row in rows
                if call["filters"].matches(row)
            }
            self.assertEqual(len(matched), 3)
            self.assertEqual(call["local_dir"], Path(temp_dir).resolve())
            manifest = Path(temp_dir) / "selection_manifest.json"
            self.assertTrue(manifest.is_file())

    def test_deleted_and_missing_zarr_rows_are_not_available(self):
        rows = make_rows(3)
        rows[0]["is_deleted"] = True
        rows[1]["zarr_processed_path"] = ""
        api, _ = make_api(rows)
        available = module.query_available_episodes(api)
        self.assertEqual([row["episode_hash"] for row in available], ["episode-02"])


if __name__ == "__main__":
    unittest.main()
