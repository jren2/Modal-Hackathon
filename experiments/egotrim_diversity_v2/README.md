# EgoTrim diversity scoring V2

This directory is an isolated, standalone experiment for measuring and curating
behavioral diversity in EgoVerse-I `fold_clothes` action cycles. It does not change
the repository's existing data, helpers, requirements, or Modal volumes. Its claims
are deliberately limited to measured composition and execution coverage; it does
not estimate or claim downstream robot-policy improvement.

## Current data status

The repository contained only its initial commit when this experiment was created;
there was no local segmentation implementation, pose helper, segment manifest, or
Modal infrastructure to reuse. The authenticated personal Modal workspace was then
inspected read-only with Modal 1.4.1:

```powershell
python -m modal volume list
python -m modal volume ls egotrim-data / --json
python -m modal volume ls egotrim-models / --json
python -m modal volume ls egoverse-zarrs-v2 /episodes --json
```

The workspace contains `egotrim-data`, `egotrim-models`, and a separately named
`egoverse-zarrs-v2` volume. On 2026-08-15, `egoverse-zarrs-v2` contained two episode
Zarrs and 188 model-ready MP4 windows under `/segments`, each with exact frame bounds
in a `manifest.json`. These are uniform fixed-window clips, not semantic action-cycle
annotations; their Zarr `annotations` arrays remain empty. The explicit EgoVerse
adapter therefore labels them with the episode task (`fold_clothes`) and limits
interpretation to within-task execution diversity.

The adapter mounts `egoverse-zarrs-v2` read-only, verifies every referenced clip and
episode store, and preserves `start_frame`/`end_frame_exclusive`. Pose extraction
uses those exact frames and the MP4's nominal uniform timeline, avoiding incorrect
time slicing across discontinuities in capture timestamps. It never silently changes
source volumes or fabricates semantic labels.

## Verified EgoVerse Zarr schema

The following schema was inspected at the exact source path
`egoverse-zarrs-v2:/episodes/2025-09-23-22-47-12-000000/`. It is a Zarr v3 group with
metadata `embodiment=human_bimanual`, `task_name=fold_clothes`, `fps=30`, and
`total_frames=2371`:

| Zarr key | Sample stored shape | Meaning |
| --- | ---: | --- |
| `left.obs_keypoints` | `(2400, 63)` | Canonical MANO joints in world coordinates |
| `right.obs_keypoints` | `(2400, 63)` | Canonical MANO joints in world coordinates |
| `left.obs_aria_keypoints` | `(2400, 63)` | Raw Aria-layout joints; reference only |
| `right.obs_aria_keypoints` | `(2400, 63)` | Raw Aria-layout joints; reference only |
| `left.obs_wrist_pose` | `(2400, 7)` | Left wrist pose in world coordinates |
| `right.obs_wrist_pose` | `(2400, 7)` | Right wrist pose in world coordinates |
| `left.obs_ee_pose` | `(2400, 7)` | Left hand end-effector pose |
| `right.obs_ee_pose` | `(2400, 7)` | Right hand end-effector pose |
| `obs_head_pose` | `(2400, 7)` | Per-frame head pose in world coordinates |
| `obs_rgb_timestamps_ns` | `(2400,)` | RGB timestamps in nanoseconds |
| `obs_eye_gaze` | `(2400, 3)` | Eye-gaze vector |
| `images.front_1` | logical `(T, 480, 640, 3)` | Per-frame JPEG imagery |
| `annotations` | `(0,)` in inspected samples | Empty variable-length byte array |

The numeric arrays are padded to a chunk boundary. The sample's `total_frames` and
nonzero timestamp count were 2371, followed by 29 all-zero rows; the median nonzero
timestamp interval was approximately 33.327 ms. The loader truncates to
`total_frames` before slicing a segment and treats remaining all-zero required joints
as invalid tracking, never as stationary motion.

Pose vectors use `[x, y, z, qw, qx, qy, qz]`. Canonical `obs_keypoints` use MANO
order: joint 0 is the wrist and joints 4, 8, 12, 16, and 20 are fingertips. Raw
`obs_aria_keypoints` have a different layout (0-4 are fingertips and 5 is the palm
root), so substituting that array would corrupt the requested feature definition.

The coordinate convention is the official EgoVerse transform:

```text
point_head[t] = inverse(head_to_world[t]) @ point_world[t]
```

The pipeline applies this inverse independently at every frame to canonical
keypoints and the left/right wrist translations. The seven-value head pose is
interpreted as translation plus a WXYZ quaternion. If a supported transform is not
present, loading fails unless the caller explicitly supplies
`--coordinate-fallback already_head_frame` after independently verifying that
convention. The fallback is recorded in outputs; it is never selected implicitly.

No source video file paths were discovered in the inspected stores. The JPEG image
array is not represented as a video path. If a segment manifest later supplies an
accessible `video_path`, it is included by reference in nearest-neighbor data and is
never copied or modified.

## Required segment manifest

Real execution requires CSV, Parquet, JSON, or JSONL records that can be mapped
unambiguously to these fields:

- `episode_id`, unique `segment_id`, and `canonical_verb` (or a supported alias);
- numeric `start_time`, `end_time`, and optionally `duration`, in episode-relative
  seconds;
- optionally paired integer `start_frame` and `end_frame_exclusive`; when present,
  these take precedence for pose slicing and are preserved in scored manifests;
- numeric `tracking_valid_frac`;
- preferably `pose_ref`, relative to `--pose-root` or absolute;
- optionally `video_path` and `source_data_path`.

Input cycles below 0.5 seconds or with declared `tracking_valid_frac < 0.7` are
rejected. If columns, timestamps, pose references, arrays, or transform semantics
cannot be mapped confidently, the process exits nonzero and writes an actionable
`schema_report.json` in the requested output directory. It does not guess fields or
manufacture data. Episode-level poses without timestamps are likewise rejected;
the uniform-timestamp fallback is allowed only for an explicitly segment-specific
pose file.

## Install and run locally

From the repository root, install only this experiment's dependencies:

```powershell
python -m pip install -r experiments/egotrim_diversity_v2/requirements_egotrim_v2.txt
```

Run the deterministic end-to-end synthetic smoke test:

```powershell
python experiments/egotrim_diversity_v2/run_egotrim_diversity_v2.py `
  --synthetic-smoke `
  --output-dir experiments/egotrim_diversity_v2/smoke_output `
  --budget-frac 0.40 `
  --seed 42
```

Run real local data after a valid manifest and pose root are available:

```powershell
python experiments/egotrim_diversity_v2/run_egotrim_diversity_v2.py `
  --segments <PATH_TO_SEGMENTS.csv> `
  --pose-root <PATH_TO_EPISODE_ZARR_ROOT> `
  --output-dir experiments/egotrim_diversity_v2/real_output `
  --budget-frac 0.40 `
  --seed 42
```

Local outputs are deliberately restricted to this experiment directory. Use a new
output subdirectory for each run to preserve earlier results.

## Feature and missing-data contract

Each accepted cycle is resampled to 30 timestamps after coordinate conversion and
bounded missing-data repair. Left and right hands remain separate. The trajectory
block is `30 x 3 x 2 = 180` values from wrist displacement relative to each hand's
first resampled wrist position. The fingertip block is
`30 x 5 x 3 x 2 = 900` fingertip-minus-same-wrist values. Log duration is appended.

Missing values are handled conservatively:

- all-zero required joints are converted to missing after Zarr padding is removed;
- measured complete-frame tracking must remain at least 0.7;
- only bounded internal runs of at most `--max-interp-gap` frames (default 2) are
  linearly interpolated;
- leading, trailing, or longer gaps are not filled;
- any required value remaining missing rejects the segment;
- substantial motion is never silently replaced with zero.

Trajectory and fingertip blocks are standardized independently, fit with separate
PCAs, and retain up to approximately 15 nondegenerate components subject to sample
count and explained variance. Standardized log duration is then appended and the
combined embedding standardized again. Fitted scalers and PCA objects are saved in
`models/`. Ablations cover trajectory-only, fingertips-only, duration-only, and the
combined representation.

## Modal entry point

`modal_app.py` is the isolated Modal runner. It refers to the existing volumes with
`create_if_missing=False`; it never creates a replacement. Source reads are allowed
from the read-only `/egotrim-data` and `/egoverse` mounts, while all new artifacts are
written beneath the unique path
`/egotrim-models/egotrim-diversity-v2/<RUN_ID>/`. Existing volume paths are never
renamed, deleted, reorganized, or overwritten.

On this workstation the `modal` console script is not on `PATH`, so use the exact
module form below. A smoke submission is limited to three episodes:

```powershell
python -m modal run experiments/egotrim_diversity_v2/modal_app.py `
  --segments /egotrim-data/<segments.csv> `
  --pose-root /egotrim-data/<pose-root> `
  --smoke `
  --run-id <RUN_ID> `
  --budget-frac 0.40 `
  --seed 42 `
  --baseline-runs 10
```

Run directly on the fixed-window model clips in `egoverse-zarrs-v2`:

```powershell
$env:PYTHONUTF8 = "1"
python -m modal run experiments/egotrim_diversity_v2/modal_app.py `
  --egoverse-clips-root /egoverse/segments `
  --egoverse-episodes-root /egoverse/episodes `
  --smoke `
  --run-id <NEW_RUN_ID> `
  --budget-frac 0.40 `
  --seed 42 `
  --baseline-runs 2
```

List its generated files without downloading or modifying them:

```powershell
python -m modal volume ls egotrim-models /egotrim-diversity-v2/<RUN_ID> --json
```

`<RUN_ID>` must be a new identifier, for example
`20260815_170000_smoke`; reusing an existing run directory is an error. The two input
modes are mutually exclusive, so use either `--segments`/`--pose-root` or the paired
EgoVerse adapter arguments.

## Outputs and interpretation

A successful run creates:

- `segment_scores.csv`, including validity, tracking quality, within-verb nearest-
  neighbor distance, distinctiveness percentile, cluster rarity, and selection;
- `episode_scores.csv`, exposing every episode-score component and rank;
- `cluster_summary.csv` and `nearest_neighbors.csv`;
- `subset_manifest.csv` and `subset_manifest.json`;
- `metrics.json`, `dashboard_data.json`, and `schema_report.json`;
- fitted artifacts under `models/` and static presentation-ready plots under
  `charts/`.

Composition diversity is reported as verb coverage and Shannon effective number of
verbs. It is not casually called Vendi. Execution diversity is computed separately
within sufficiently sampled verbs using median-bandwidth RBF kernels and Vendi
scores; undersampled verbs are reported separately. Segment distinctiveness is the
within-verb k-nearest-neighbor distance percentile, with cluster rarity separate.
Leave-one-out Vendi is optional and explicitly labelled a non-monotonic diagnostic.

The default curator uses monotonic facility-location coverage with sufficiently
populated verb preservation under both count and duration limits. Comparisons at
identical budgets include uniform random, verb-stratified random, annotation-only,
and cluster-medoid baselines; random results are summarized across deterministic
seeds. These metrics support only the statement that one subset retained more or
less measured behavioral coverage than another on the evaluated representation.
They do not support claims about policy quality, task success, or deployment
performance.

Run tests with:

```powershell
python -m pytest experiments/egotrim_diversity_v2/test_egotrim_diversity_v2.py -q
```
