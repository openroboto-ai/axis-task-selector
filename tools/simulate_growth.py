"""模拟首期 30、每期新增 10，检查编号、任务组、累计覆盖和耗尽边界。"""

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selector import draw, load_baseline_ids, load_pool, pool_fingerprint  # noqa: E402


def main():
    pool = load_pool()
    tasks = {t["task_id"]: t for t in pool["tasks"]}
    baseline = load_baseline_ids()

    def round_summary(index, history):
        sampled = [tasks[tid] for tid in history if tid in tasks]
        return {
            "round": index,
            "total_tasks": len(history),
            "in_pool_tasks": len(sampled),
            "difficulty": dict(collections.Counter(task["difficulty"] for task in sampled)),
            "categories": dict(collections.Counter(task["category"] for task in sampled)),
        }

    reports = []
    for base_seed in range(16):
        history, batches = baseline.copy(), []
        rounds = [round_summary(1, history)]
        stop_reason = None
        while True:
            seed = base_seed * 10000 + len(batches)
            before = history.copy()
            try:
                added = draw(seed, len(history), history, pool)
            except ValueError as exc:
                if "题库耗尽" not in str(exc) and "任务组不足" not in str(exc):
                    raise
                stop_reason = str(exc)
                break
            assert len(added) == len(set(added)) == 10
            assert not set(added) & set(history)
            assert len({tasks[tid]["task_group"] for tid in added}) == 10
            assert added == draw(seed, len(history), list(reversed(history)), pool)
            history.extend(added)
            assert history[: len(before)] == before
            batches.append({"seed": seed, "added_ids": added})
            assert len(history) == 30 + 10 * len(batches)
            rounds.append(round_summary(len(batches) + 1, history))
        remaining = set(tasks) - set(history)
        groups = {tasks[tid]["task_group"] for tid in remaining}
        assert len(remaining) < 10 or len(groups) < 10
        reports.append(
            {
                "base_seed": base_seed,
                "completed_rounds": len(rounds),
                "selected_tasks": len(history),
                "remaining_tasks": len(remaining),
                "remaining_groups": len(groups),
                "stop_reason": stop_reason,
                "batches": batches,
                "rounds": rounds,
            }
        )
    result = {
        "pool_sha256": pool_fingerprint(pool),
        "initial_task_ids": baseline,
        "tasks": len(tasks),
        "task_groups": len({t["task_group"] for t in tasks.values()}),
        "simulations": len(reports),
        "duplicate_ids": 0,
        "within_batch_group_conflicts": 0,
        "growth_rule_violations": 0,
        "runs": reports,
    }
    (ROOT / "evidence/growth_simulation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "任务数": len(tasks),
                "任务组": result["task_groups"],
                "模拟次数": len(reports),
                "完成期数范围": [
                    min(r["completed_rounds"] for r in reports),
                    max(r["completed_rounds"] for r in reports),
                ],
                "最终已选数量范围": [
                    min(r["selected_tasks"] for r in reports),
                    max(r["selected_tasks"] for r in reports),
                ],
                "剩余任务组范围": [
                    min(r["remaining_groups"] for r in reports),
                    max(r["remaining_groups"] for r in reports),
                ],
                "重复编号": 0,
                "批内同组冲突": 0,
                "增长规律错误": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
