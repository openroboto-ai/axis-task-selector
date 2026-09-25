"""编号、任务组硬约束、累计增长与 AXIS 原始格式回归测试。"""

import copy
import csv
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selector import (  # noqa: E402
    draw,
    load_baseline_ids,
    load_pool,
    normalize_seed,
    official_task_id,
    pool_fingerprint,
    public_task_ids,
    read_axis_metadata,
)


def fixture_pool():
    tasks = []
    for group in range(30):
        for difficulty in ("simple", "constrained", "compound"):
            tasks.append(
                {
                    "task_id": len(tasks) + 1,
                    "category": str(group % 3),
                    "difficulty": difficulty,
                    "task_group": f"g{group}",
                }
            )
    return {"profile": "test", "tasks": tasks}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.pool = fixture_pool()

    def test_replay_and_input_order_independence(self):
        history = [1, 2, 4, 8]
        expected = draw(42, 4, history, self.pool)
        reversed_pool = {**self.pool, "tasks": list(reversed(self.pool["tasks"]))}
        self.assertEqual(expected, draw("0x2a", 4, list(reversed(history)), reversed_pool))
        self.assertEqual(pool_fingerprint(self.pool), pool_fingerprint(reversed_pool))
        self.assertNotEqual(expected, draw(99, 4, history, self.pool))

    def test_one_task_per_group_is_hard_constraint(self):
        # 同组中即使有多种难度且平衡收益更高，也不能在本批重复选择。
        lookup = {t["task_id"]: t for t in self.pool["tasks"]}
        for seed in range(30):
            chosen = draw(seed, 0, [], self.pool)
            self.assertEqual(len({lookup[tid]["task_group"] for tid in chosen}), 10)

    def test_groups_can_reappear_in_later_batches_but_ids_cannot(self):
        pool = {
            "tasks": [
                {"task_id": i + 1, "category": "one", "difficulty": "one", "task_group": f"g{i % 10}"}
                for i in range(20)
            ]
        }
        first = draw(1, 0, [], pool)
        second = draw(2, 10, first, pool)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(set(first + second), set(range(1, 21)))

    def test_ten_tasks_in_fewer_than_ten_groups_must_fail(self):
        pool = {
            "tasks": [
                {"task_id": i + 1, "category": "one", "difficulty": "one", "task_group": f"g{i % 9}"} for i in range(90)
            ]
        }
        with self.assertRaisesRegex(ValueError, "任务组不足"):
            draw(0, 0, [], pool)

    def test_no_partial_batch_near_exhaustion(self):
        with self.assertRaisesRegex(ValueError, "题库耗尽"):
            draw(1, 81, list(range(1, 82)), self.pool)

    def test_invalid_inputs_and_immutable_arguments(self):
        before = copy.deepcopy(self.pool)
        for seed, count, history in [
            (True, 0, []),
            (-1, 0, []),
            (2**256, 0, []),
            (0, True, []),
            (0, 1, []),
            (0, 2, [1, 1]),
            (0, 1, [99999]),
            (0, 1, [True]),
            (0, 1, ["1"]),
        ]:
            with self.subTest(seed=seed, count=count, history=history):
                with self.assertRaises(ValueError):
                    draw(seed, count, history, self.pool)
        history = [1]
        draw(5, 1, history, self.pool)
        self.assertEqual(history, [1])
        self.assertEqual(self.pool, before)
        self.assertEqual(normalize_seed(2**256 - 1), "f" * 64)

    def test_historical_distribution_is_considered(self):
        pool = {
            "tasks": [
                {"task_id": i + 1, "category": str(i // 40), "difficulty": str(i // 40), "task_group": str(i)}
                for i in range(120)
            ]
        }
        chosen = draw(11, 12, list(range(1, 13)), pool)
        self.assertTrue(all(tid > 40 for tid in chosen))

    def test_baseline_ids_are_history_only_and_unknown_ids_still_fail(self):
        pool = {**self.pool, "baseline_task_ids": [1001, 1002]}
        history = [1001, 1002, 1]
        added = draw(42, 3, history, pool)
        self.assertTrue(set(added) <= {t["task_id"] for t in pool["tasks"]})
        self.assertFalse(set(added) & set(history))
        reordered = {**pool, "baseline_task_ids": [1002, 1001]}
        self.assertEqual(added, draw(42, 3, list(reversed(history)), reordered))
        self.assertEqual(pool_fingerprint(pool), pool_fingerprint(reordered))
        with self.assertRaisesRegex(ValueError, "已选编号"):
            draw(42, 1, [1003], pool)
        for invalid in ([True], [1001, 1001], [0], "1001"):
            with self.subTest(baseline=invalid), self.assertRaisesRegex(ValueError, "首期任务编号"):
                draw(42, 0, [], {**pool, "baseline_task_ids": invalid})


class AxisFormatTests(unittest.TestCase):
    def test_official_directory_ids(self):
        self.assertEqual(official_task_id("archives/task_638_isaaclab_state_replay_100.zarr"), 638)
        self.assertEqual(official_task_id("Franka-800-put_the_scissors_on_the_gift_box"), 800)
        self.assertIsNone(official_task_id("metadata.json"))

    def test_v1_and_v2_metadata_are_preserved_without_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            v1 = {
                "task_name": "task_501_isaac_state_train",
                "joint_pos_padding": 9,
                "observation_space": "joint_pos",
                "action_space": "joint_pos",
                "num_episodes": 100,
            }
            v2 = {
                "schema": "axis_franka_task_upload_manifest_v2",
                "task_id": 800,
                "task_name": "Put the Scissors on the Gift Box",
                "embodiment_id": "franka",
                "episode_attempt_mapping": {"fixed": []},
            }
            p1 = root / "task_501_isaac_state_train_100.zarr/metadata.json"
            p2 = root / "Franka-800-put_the_scissors_on_the_gift_box/task_manifest.json"
            write_json(p1, v1)
            write_json(p2, v2)
            before = (p1.read_bytes(), p2.read_bytes())
            self.assertEqual(read_axis_metadata(root), {501: [v1], 800: [v2]})
            self.assertEqual(before, (p1.read_bytes(), p2.read_bytes()))

    def test_manifest_id_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            write_json(
                root / "Franka-800-example/task_manifest.json",
                {"task_id": 801, "task_name": "example", "embodiment_id": "franka"},
            )
            with self.assertRaisesRegex(ValueError, "不一致"):
                read_axis_metadata(root)

    def test_local_datasets_only_select_annotated_tasks_actually_present(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for task_id in (501, 99999):
                write_json(
                    root / f"Franka-{task_id}-example/task_manifest.json",
                    {
                        "task_id": task_id,
                        "task_name": "example",
                        "embodiment_id": "franka",
                    },
                )
            self.assertEqual([t["task_id"] for t in load_pool([root])["tasks"]], [501])


class DeliveryTests(unittest.TestCase):
    def test_annotation_table_matches_the_public_sampling_pool(self):
        with (ROOT / "data/annotations.csv").open(encoding="utf-8", newline="") as stream:
            annotations = list(csv.DictReader(stream))
        ids = {int(row["task_id"]) for row in annotations}
        self.assertEqual(len(annotations), 1233)
        self.assertEqual(ids, {t["task_id"] for t in load_pool()["tasks"]})
        self.assertLessEqual(ids, public_task_ids())

    def test_shipped_pool_and_vectors(self):
        pool = load_pool()
        self.assertEqual(len(pool["tasks"]), 1233)
        self.assertEqual(len({t["task_group"] for t in pool["tasks"]}), 210)
        for vector in json.loads((ROOT / "tests/vectors.json").read_text()):
            self.assertEqual(vector["pool_sha256"], pool_fingerprint(pool))
            self.assertEqual(draw(**vector["input"], pool=pool), vector["output"])

    def test_published_thirty_then_weekly_ten_growth_for_50_rounds(self):
        pool = load_pool()
        lookup = {t["task_id"]: t for t in pool["tasks"]}
        history = load_baseline_ids()
        self.assertEqual(len(history), 30)
        for batch in range(50):
            added = draw(20260923 + batch, len(history), history, pool)
            self.assertEqual(len(added), 10)
            self.assertFalse(set(added) & set(history))
            self.assertEqual(len({lookup[tid]["task_group"] for tid in added}), 10)
            old = history.copy()
            history.extend(added)
            self.assertEqual(history[: len(old)], old)
            self.assertEqual(len(history), 30 + 10 * (batch + 1))
        self.assertEqual(len(history), 530)

    def test_published_baseline_matches_docs_and_cli_reproduces_first_addition(self):
        baseline = load_baseline_ids()
        self.assertEqual(len(baseline), 30)
        for filename in ("README.md",):
            document = (ROOT / "task-sets/axis_v1.0" / filename).read_text()
            self.assertEqual([int(tid) for tid in re.findall(r"^\| (\d+) \|", document, re.MULTILINE)], baseline)
        request = json.loads((ROOT / "examples/next10_request.json").read_text())
        self.assertEqual(request["selected_ids"], baseline)
        self.assertEqual(request["current_count"], 30)
        pool_ids = {t["task_id"] for t in load_pool()["tasks"]}
        self.assertEqual(len(set(baseline) - pool_ids), 22)
        process = subprocess.run(
            [sys.executable, str(ROOT / "selector.py")],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            cwd="/tmp",
            check=True,
        )
        added = json.loads(process.stdout)
        self.assertEqual(added, json.loads((ROOT / "examples/next10_response.json").read_text()))
        self.assertEqual(added, draw(**request))
        self.assertFalse(set(added) & set(baseline))
        self.assertLessEqual(set(added), pool_ids)

    def test_cli_outside_repo_and_atomic_error(self):
        request = {"seed": "0x1234", "current_count": 0, "selected_ids": []}
        result = subprocess.run(
            [sys.executable, str(ROOT / "selector.py")],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            cwd="/tmp",
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), draw(**request))
        result = subprocess.run(
            [sys.executable, str(ROOT / "selector.py")], input='{"seed":0}', capture_output=True, text=True, cwd="/tmp"
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
