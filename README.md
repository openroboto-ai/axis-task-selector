# AXIS Task Selector

Given a **seed**, the **number of previously selected tasks**, and their **complete ID list**, select **10 new official AXIS task IDs**. Requires Python 3.10+ and uses only the standard library. Default selection works offline.

## Published task set

| Benchmark | Tasks | Documentation |
|---|---:|---|
| `axis_v1.0` | 30 | [README](task-sets/axis_v1.0/README.md) |

Each task links directly to its official AXIS Hub page.

## Sampling pool and data sources

The current sampling pool contains **1,233 tasks**, **210 task groups**, and **8 task categories**. IDs come from AXIS's published Hugging Face directories and are matched to the supplied sampling annotations.

| Source | Official repository | Metadata format |
|---|---|---|
| V1 | [Franka-Dataset](https://huggingface.co/datasets/axisrobotics/Franka-Dataset) | `task_<id>_...zarr/metadata.json` |
| V2 | [Franka-Datasets-v2](https://huggingface.co/datasets/axisrobotics/Franka-Datasets-v2), [Franka-dataset-v2-2](https://huggingface.co/datasets/axisrobotics/Franka-dataset-v2-2) | `Franka-<id>-<task-name>/task_manifest.json` |
| V2 variants | [5k](https://huggingface.co/datasets/axisrobotics/Franka-Datasets-v2-5k), [30k-LIBERO](https://huggingface.co/datasets/axisrobotics/Franka-Datasets-v2-30k-LIBERO), [MultiEmbodiment](https://huggingface.co/datasets/axisrobotics/AXIS-Datasets-v2-MultiEmbodiment) | Franka entries, deduplicated by official ID |

Pinned source revisions are in [data/sources.json](data/sources.json). [data/hf](data/hf) contains snapshots of official file listings, so selecting IDs does not require downloading trajectories. Local datasets are read in their original AXIS V1/V2 formats.

[data/annotations.csv](data/annotations.csv) supplies `category`, `difficulty`, `task_group`, and `label_basis`, joined by official `task_id`. These are **project-provided sampling annotations, not official AXIS labels**.

## Task groups and structural difficulty

A **task group** contains tasks with the same scene definition and initial state; their objectives can differ. Group membership is based on those definitions, not task names, and is fixed in the annotation table.

**Each batch of 10 tasks contains 10 distinct task groups.** A later batch may select another task from a previously used group, but no task ID can repeat across the complete selection history.

Difficulty describes task structure:

| Level | Meaning |
|---|---|
| `simple` | One primary goal without the additional constraints below |
| `constrained` | One goal involving orientation, joints, containment/insertion, or stacking |
| `compound` | At least two non-gripper goal predicates must hold together; alternatives in an OR condition do not count as multiple goals |

These labels describe structural complexity, not measured model success rates. The eight categories are pick/lift, relative placement, containment/insertion, stacking/balance, rotation/orientation, rearrangement/sorting, target pose, and push/pull displacement.

## Input and output

Input contains exactly three fields. This example uses the complete published `axis_v1.0` baseline and a demonstration seed:

```json
{
  "seed": "0x20260930",
  "current_count": 30,
  "selected_ids": [22,31,33,34,35,37,40,41,42,43,44,46,48,49,50,51,52,53,54,55,56,57,501,502,503,504,505,506,514,757]
}
```

- `seed`: an unsigned 256-bit integer or a hexadecimal string beginning with `0x`. Use a string when passing large seeds through JavaScript.
- `current_count`: the number of tasks already selected; it must equal the length of `selected_ids`.
- `selected_ids`: the complete history, including the initial 30 tasks and every subsequent addition, not just the previous batch.

The example returns:

```json
[675,1720,1455,892,868,1153,1060,1605,2308,764]
```

This matches [examples/next10_request.json](examples/next10_request.json) and [examples/next10_response.json](examples/next10_response.json). Append the 10 IDs and submit `current_count=40` with all 40 IDs on the next call. The caller maintains the history and supplies the seed; this seed is for reproduction examples only.

The fixed initial 30 IDs are listed in [task_ids.json](task-sets/axis_v1.0/task_ids.json). Twenty-two are outside the sampling pool and are accepted as historical inputs by the selector; do not remove them or change `current_count` to 8. They participate in the complete-history hash; coverage balancing counts annotated in-pool history. New tasks always come from the 1,233-task sampling pool. Other unknown IDs remain errors.

## Selection algorithm

Algorithm: `axis-balanced-ten-v3`.

1. Validate the input and remove previously selected task IDs.
2. Require at least 10 remaining tasks from at least 10 distinct groups.
3. Favor categories and difficulty levels with lower cumulative coverage, counting in-pool history and picks already made in this batch.
4. Break ties by preferring less frequently selected groups, then by a SHA-256 ordering derived from the seed, pool hash, history, and task ID.
5. After each pick, update the counts and exclude other tasks in its group for this batch. Repeat until 10 tasks are selected.

The coverage score in step 3 is minimized:

```text
number of categories × (2 × selected count in this category + 1)
+ number of difficulty levels × (2 × selected count at this difficulty + 1)
```

Selection favors balanced cumulative coverage. Category sizes and group constraints limit the achievable balance; equal category counts in every batch and equal probability for every task are not guaranteed.

For the same seed, full history, and pool/annotation version, the output is identical. Fix the repository revision when reproducing a draw; reference input/output pairs are in [tests/vectors.json](tests/vectors.json).

## Usage

Run from the repository root:

```bash
git clone https://github.com/OpenRoboto/axis-task-selector.git
cd axis-task-selector

# Select from the supplied pool.
python selector.py < examples/empty_history.json

# Add 10 tasks to the published axis_v1.0 baseline and save the result for handoff.
python selector.py < examples/next10_request.json > /tmp/axis-new-ids.json
cat /tmp/axis-new-ids.json

# Restrict selection to annotated tasks present in local AXIS datasets.
python selector.py \
  --dataset-root /path/to/Franka-Dataset \
  --dataset-root /path/to/Franka-Datasets-v2 \
  < examples/empty_history.json
```

When local directories are supplied, the pool is their intersection with the annotation table. Different subsets produce different pool hashes. Except for the fixed baseline IDs, all historical IDs must belong to the current pool.

Pass the output JSON unchanged to the evaluator sync script with `--selection /tmp/axis-new-ids.json`, using the same seed and previous benchmark. The sync script verifies replay before generating the next YAML; see the public [integration script](https://github.com/openroboto-ai/openroboto-evaluation/blob/main/tools/sync_axis_benchmark.py) and its `--help` output.

Python API:

```python
from selector import draw, load_baseline_ids

history = load_baseline_ids()
new_ids = draw(seed="0x20260930", current_count=len(history), selected_ids=history)
```

## Errors and exhaustion

Invalid seeds, duplicate or unknown IDs, inconsistent counts, and metadata/directory ID mismatches produce errors. The CLI exits with status 2 and writes no partial selection to stdout.

Selection also stops when fewer than 10 tasks or 10 distinct groups remain. The distinct-group rule is never relaxed to fill a batch.

## Tests and growth simulation

```bash
python -m unittest discover -s tests -v
python tools/simulate_growth.py
```

Tests cover original metadata loading, input validation, deterministic replay, history preservation, task-group constraints and cumulative growth.

The [growth simulation](evidence/growth_simulation.json) starts with the actual published `axis_v1.0` baseline and adds 10 per round. All 16 seed sequences completed **122 rounds including the baseline: 121 additions, 1,240 total tasks**. There were no duplicate IDs, within-batch group conflicts or growth-rule violations. The remaining 15 pool tasks occupy 8 groups, so the next batch correctly stops.
