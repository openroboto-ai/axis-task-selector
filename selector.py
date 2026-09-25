#!/usr/bin/env python3
"""从 AXIS 官方编号中确定性抽取 10 题，同一批内任务组不重复。"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import re
import sys

ALGORITHM = "axis-balanced-ten-v3"
BATCH_SIZE = 10
ROOT = pathlib.Path(__file__).resolve().parent


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def normalize_seed(seed):
    if isinstance(seed, str) and re.fullmatch(r"0x[0-9a-fA-F]{1,64}", seed):
        seed = int(seed, 16)
    if type(seed) is not int or not 0 <= seed < 2**256:
        raise ValueError("seed 必须是无符号 256 位整数或 0x 开头的十六进制字符串")
    return f"{seed:064x}"


def official_task_id(path):
    """从 AXIS 原始 V1/V2 目录名称读取编号，保留官方编号。"""
    name = pathlib.PurePosixPath(path).name
    match = re.match(r"task_(\d+)_.*\.zarr(?:$|\.)", name)
    if match is None:
        match = re.fullmatch(r"(?:Franka|Sawyer)-(\d+)-.+", name)
    return int(match[1]) if match else None


def read_axis_metadata(root):
    """原样读取 V1 metadata.json / V2 task_manifest.json，不转换轨迹格式。"""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise ValueError(f"数据目录不存在：{root}")
    tasks = collections.defaultdict(list)
    for pattern in ("metadata.json", "task_manifest.json"):
        for path in sorted(root.rglob(pattern)):
            task_id = official_task_id(path.parent.name)
            if task_id is None:
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"AXIS 元数据必须是对象：{path}")
            if pattern == "task_manifest.json":
                if type(metadata.get("task_id")) is not int or metadata["task_id"] != task_id:
                    raise ValueError(f"目录编号与 task_manifest.task_id 不一致：{path}")
                if not isinstance(metadata.get("task_name"), str) or not metadata["task_name"].strip():
                    raise ValueError(f"task_manifest 缺少 task_name：{path}")
                if metadata.get("embodiment_id") != "franka":
                    continue
            else:
                name = metadata.get("task_name", "")
                if not isinstance(name, str) or not re.match(rf"task_{task_id}_", name):
                    raise ValueError(f"V1 metadata.task_name 与目录编号不一致：{path}")
                if metadata.get("joint_pos_padding") != 9:
                    raise ValueError(f"V1 数据不是本版本的 9 维接口：{path}")
            tasks[task_id].append(metadata)
    if not tasks:
        raise ValueError("目录内未找到受支持的 AXIS V1/V2 原始任务元数据")
    return dict(tasks)


def public_task_ids():
    """离线读取固定 revision 的官方 HF 原始文件目录响应。"""
    sources = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    result = set()
    for source in sources:
        entries = json.loads((ROOT / source["listing"]).read_text(encoding="utf-8"))
        for entry in entries:
            task_id = official_task_id(entry["path"])
            if task_id is not None and not pathlib.PurePosixPath(entry["path"]).name.startswith("Sawyer-"):
                result.add(task_id)
    return result


def validate_pool(pool):
    if not isinstance(pool, dict) or not isinstance(pool.get("tasks"), list) or not pool["tasks"]:
        raise ValueError("题库必须包含任务")
    by_id = {}
    for task in pool["tasks"]:
        if not isinstance(task, dict):
            raise ValueError("任务条目必须是对象")
        tid = task.get("task_id")
        if type(tid) is not int or tid <= 0 or tid in by_id:
            raise ValueError(f"无效或重复的官方 task ID：{tid!r}")
        for field in ("category", "difficulty", "task_group"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"任务 {tid} 缺少抽样标注：{field}")
        by_id[tid] = task
    baseline = pool.get("baseline_task_ids", [])
    if (
        not isinstance(baseline, list)
        or any(type(tid) is not int or tid <= 0 for tid in baseline)
        or len(set(baseline)) != len(baseline)
    ):
        raise ValueError("首期任务编号必须为不重复的正整数数组")
    return by_id


def load_baseline_ids():
    """已公布的 axis_v1.0 完整题单；池外编号只允许出现在历史中。"""
    return json.loads((ROOT / "task-sets/axis_v1.0/task_ids.json").read_text(encoding="utf-8"))


def load_pool(dataset_roots=None):
    """从公开目录或原始数据目录加载，CSV 仅保存额外抽样标注。"""
    available = public_task_ids()
    if dataset_roots:
        available = set()
        for root in dataset_roots:
            available.update(read_axis_metadata(root))
    with (ROOT / "data/annotations.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tasks, seen = [], set()
    for row in rows:
        tid = int(row["task_id"])
        if tid in seen:
            raise ValueError(f"抽样标注内编号重复：{tid}")
        seen.add(tid)
        if tid in available:
            tasks.append({"task_id": tid, **{key: row[key] for key in ("category", "difficulty", "task_group")}})
    pool = {
        "profile": "released",
        "annotation_version": "axis-structural-v1",
        "baseline_task_ids": load_baseline_ids(),
        "tasks": sorted(tasks, key=lambda t: t["task_id"]),
    }
    validate_pool(pool)
    return pool


def pool_fingerprint(pool):
    normalized = {**pool, "tasks": sorted(pool["tasks"], key=lambda t: t["task_id"])}
    if "baseline_task_ids" in pool:
        normalized["baseline_task_ids"] = sorted(pool["baseline_task_ids"])
    return canonical_hash(normalized)


def draw(seed, current_count, selected_ids, pool=None):
    """返回恰好 10 个新编号；全局不重复，同一批来自 10 个不同任务组。"""
    seed = normalize_seed(seed)
    pool = load_pool() if pool is None else pool
    by_id = validate_pool(pool)
    if type(current_count) is not int or current_count < 0:
        raise ValueError("current_count 必须是非负整数")
    if not isinstance(selected_ids, list) or any(type(tid) is not int for tid in selected_ids):
        raise ValueError("selected_ids 必须是整数编号数组")
    selected = set(selected_ids)
    if len(selected) != len(selected_ids):
        raise ValueError("selected_ids 有重复编号")
    if current_count != len(selected_ids):
        raise ValueError("current_count 必须等于 selected_ids 的长度")
    unknown = selected - by_id.keys() - set(pool.get("baseline_task_ids", []))
    if unknown:
        raise ValueError(f"已选编号不在当前抽样池或已公布首期题单内：{sorted(unknown)}")
    remaining = {tid: task for tid, task in by_id.items() if tid not in selected}
    if len(remaining) < BATCH_SIZE:
        raise ValueError(f"题库耗尽：需要 10 个新任务，实际剩余 {len(remaining)} 个")
    group_count = len({task["task_group"] for task in remaining.values()})
    if group_count < BATCH_SIZE:
        raise ValueError(f"任务组不足：需要 10 个不同任务组，实际剩余 {group_count} 组；不会放宽同组互斥规则")
    categories = {task["category"] for task in by_id.values()}
    difficulties = {task["difficulty"] for task in by_id.values()}
    # 池外首期题属于完整历史并参与 seed 排序，但没有本抽样池的覆盖标注。
    sampled_history = selected & by_id.keys()
    category_count = collections.Counter(by_id[tid]["category"] for tid in sampled_history)
    difficulty_count = collections.Counter(by_id[tid]["difficulty"] for tid in sampled_history)
    history_groups = collections.Counter(by_id[tid]["task_group"] for tid in sampled_history)
    context = canonical_hash(
        {"algorithm": ALGORITHM, "pool_sha256": pool_fingerprint(pool), "seed": seed, "selected_ids": sorted(selected)}
    )
    random_rank = {tid: hashlib.sha256(f"{context}:{tid}".encode("ascii")).digest() for tid in remaining}
    result = []
    for _ in range(BATCH_SIZE):

        def priority(tid):
            task = remaining[tid]
            imbalance = len(categories) * (2 * category_count[task["category"]] + 1) + len(difficulties) * (
                2 * difficulty_count[task["difficulty"]] + 1
            )
            return imbalance, history_groups[task["task_group"]], random_rank[tid], tid

        tid = min(remaining, key=priority)
        task = remaining[tid]
        result.append(tid)
        category_count[task["category"]] += 1
        difficulty_count[task["difficulty"]] += 1
        history_groups[task["task_group"]] += 1
        # 本次选中一组后，排除本次所有同组任务；其他批次仍可选择该组未选过的题。
        remaining = {key: value for key, value in remaining.items() if value["task_group"] != task["task_group"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=pathlib.Path, action="append", help="AXIS V1/V2 原始数据目录，可指定多次"
    )
    args = parser.parse_args()
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict) or set(request) != {"seed", "current_count", "selected_ids"}:
            raise ValueError("输入必须且只能包含 seed、current_count、selected_ids")
        result = draw(**request, pool=load_pool(args.dataset_root))
    except (ValueError, TypeError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
