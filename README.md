# Download EgoVerse into Modal

Each person downloads the dataset into a Volume owned by their Modal workspace.
This GitHub repository contains the downloader code—not the Zarr data or
credentials.

## 1. Clone the repositories

```bash
git clone https://github.com/GaTech-RL2/EgoVerse.git
```

EgoVerse is needed locally to run `setup_secret.sh`. The Modal container also
clones EgoVerse while building its image.

## 2. Install the command-line tools

On macOS:

```bash
brew install awscli
python3.11 -m pip install --upgrade modal
```

Verify the installations and authenticate with Modal:

```bash
aws --version
modal --version
modal setup
```

The active Modal account and workspace determine where the Volume is created.

## 3. Configure authorized AWS credentials

Obtain your own authorized AWS bootstrap credentials from the EgoVerse/RL2
team, then run:

```bash
aws configure
```

Enter:

- Your AWS access-key ID
- Your AWS secret-access key
- Region: `us-east-2`
- Output format: leave blank or use `json`

Never use credentials committed to GitHub or shared in chat. Verify access
without exposing the credentials:

```bash
aws sts get-caller-identity
```

## 4. Generate the protected EgoVerse environment

From the cloned EgoVerse repository:

```bash
cd ../EgoVerse

ENV_FILE=/private/tmp/egoverse_env \
  bash egomimic/utils/aws/setup_secret.sh
```

This retrieves the read-only registry and R2 configuration used by the official
downloader. Confirm that the protected file exists without printing it:

```bash
test -f /private/tmp/egoverse_env && echo "EgoVerse environment ready"
```

Do not commit or print `/private/tmp/egoverse_env`.

## 5. Download the dataset into Modal

Return to this repository:

```bash
cd ../<your-modal-repo>

eval "$(aws configure export-credentials --format env)"
modal run modal_official_sync.py
```

The Modal job runs the equivalent of:

```bash
python egomimic/scripts/data_download/sync_s3.py \
  --local-dir /egoverse/episodes \
  --filters aria-fold-clothes \
  --workers 32
```

It automatically creates this Volume in your active Modal workspace:

```text
egoverse-zarrs-v2
└── episodes/
    ├── <episode-id>/
    ├── <episode-id>/
    └── ...
```

The download is resumable. If the local command disconnects or times out, run
the same commands again:

```bash
eval "$(aws configure export-credentials --format env)"
modal run modal_official_sync.py
```

The downloader revisits incomplete episodes while skipping objects already
present and matching the source.

## 6. Verify access from a Modal container

Run the lightweight proof-of-access script:

```bash
modal run modal_inspect_volume.py
```

It mounts the Volume in a Modal container and reads one real episode metadata
file and one physical image chunk.

You can also browse the Volume using the Modal CLI:

```bash
modal volume ls egoverse-zarrs-v2 episodes
```

To recursively count every stored file:

```bash
modal run modal_inspect_volume.py --full-count
```

## Use the Volume in another Modal function

Members of the same Modal workspace can mount the existing Volume by name:

```python
import modal

app = modal.App("my-egoverse-job")
volume = modal.Volume.from_name("egoverse-zarrs-v2", version=2)


@app.function(volumes={"/egoverse": volume})
def process_episodes():
    episodes_path = "/egoverse/episodes"
    print(episodes_path)
```

EgoVerse episodes are Zarr datasets. Image streams, annotations, poses, and
timestamps are stored as Zarr groups and arrays rather than necessarily as one
ordinary `.mp4` file per episode.

## Sharing behavior

- Members of the same Modal workspace can all mount `egoverse-zarrs-v2`. Only
  one download is necessary.
- Different Modal accounts or workspaces must each run the downloader and
  create their own copy.
- Cloning this GitHub repository does not transfer the Volume data.
- To distribute data across unrelated workspaces, use an authorized shared
  S3/R2 bucket as the canonical source and let each workspace read or ingest
  from it.

Never commit AWS credentials, generated dotenv files, or downloaded Zarr data
to GitHub.

## Browse kinematic segments

Launch a small web UI that reads the manifests and representative camera frames
directly from the Volume:

```bash
# Temporary development URL (reloads when this file changes)
modal serve modal_segment_browser.py

# Persistent deployed URL
modal deploy modal_segment_browser.py
```

The browser can switch between episodes and between kinematic and fixed
one-second boundaries. It provides a video player with a synchronized segment
timeline; clicking a segment seeks the video to that boundary. This small demo
uses a public Modal endpoint, so anyone with its URL can view the footage.

## Create one-second MP4 segments

The source episodes store JPEG frames in Zarr arrays. Create playable one-second
H.264 clips from the `images.front_1` stream with:

```bash
# Safe smoke test: process one episode
modal run modal_segment_videos.py

# Process ten episodes in parallel
modal run modal_segment_videos.py --max-episodes 10

# Process every complete episode
modal run modal_segment_videos.py --max-episodes 0
```

Outputs are written back to the same Volume without changing the source data:

```text
segments/<episode-id>/front_1/
├── 000000.mp4
├── 000001.mp4
├── ...
└── manifest.json
```

Each manifest records the source frame range, start time, and duration of every
clip. The last clip can be shorter than one second when an episode does not end
on an exact one-second boundary. Completed episodes are skipped on subsequent
runs; pass `--overwrite` to regenerate them.

## Segment two episodes

To segment the first two available EgoVerse episodes into one-second MP4 clips:

```bash
modal run modal_segment_videos.py --max-episodes 2
```

The command is safe to run again. Episodes with complete outputs are verified
and skipped, while missing or incomplete outputs are resumed. The clips and
their manifests are available to functions in the same Modal workspace at:

```text
/egoverse/segments/<episode-id>/front_1/
```

## Kinematic segmentation

Generate motion-driven boundaries and a fixed one-second baseline for the first
two episodes:

```bash
modal run modal_kinematic_segment.py --max-episodes 2
```

The kinematic method uses head-relative left/right hand poses, smoothed and
robustly normalized hand linear speed, hand angular speed, and inter-hand
distance. PELT proposes only sustained change points. The script merges segments
shorter than 1.5 seconds and normally splits segments longer than 4 seconds,
with a small tolerance for coherent phrases. Annotations are attached by overlap
after boundary detection; they do not influence the boundaries.

Results are JSON manifests in the existing Volume:

```text
/egoverse/kinematic_segments/<episode-id>/
├── kinematic.json
└── fixed_1s.json
```

Each manifest contains ordered, end-exclusive frame ranges:

```json
{
  "start_idx": 42,
  "end_idx": 71,
  "start_time": 1.4,
  "end_time": 2.3667,
  "annotation": "folding the left sleeve"
}
```

The defaults can be tuned from the command line:

```bash
modal run modal_kinematic_segment.py \
  --max-episodes 2 \
  --minimum-seconds 1.5 \
  --maximum-seconds 4.0 \
  --smoothing-seconds 0.25 \
  --penalty 120 \
  --head-weight 0.0 \
  --mode both
```

Use `--mode kinematic` or `--mode fixed` to generate only one method. A lower
PELT penalty produces more boundaries; a higher penalty produces fewer.

## Experimental task-attempt extraction

The current project goal is to extract complete demonstrations from raw
episodes—not to treat short kinematic phrases as final comparison units. Run the
V1 attempt classifier on one episode with:

```bash
modal run modal_extract_attempts.py --max-episodes 1
```

It samples the center RGB frame from each one-second window and makes one
`Qwen/Qwen2.5-VL-3B-Instruct` scene decision. A visible task object/workspace is
a `TASK` candidate; a missing object, prominent unrelated person, setup,
cleanup, or empty scene is a hard `IRRELEVANT` cut. Annotation overlap and
head-relative hand activity provide lightweight supporting evidence. Predictions
are temporally cleaned and saved with candidate attempt ranges at:

```text
/egoverse/attempts/<episode-id>/attempts.json
```

The manifest retains every raw/cleaned window prediction, confidence, sampled
frame indices, VLM reason, annotation evidence, and hand-activity score.

### Current validation status

This fast V1 prioritizes removing obvious irrelevant footage and producing a
running attempt-candidate pipeline. It does not try to infer subtle fold versus
unfold direction from short temporal context. Manifests include
`needs_human_review` and classifier-collapse warnings so questionable episodes
can be inspected before automatic keep/drop decisions.

## Attempt-level physical features and similarity

After `modal_extract_attempts.py` has created attempt manifests, extract
head-relative physical features and compare attempts within each task:

```bash
# Smoke test the first episode with an attempt manifest
modal run modal_attempt_features.py --max-episodes 1

# Process every available attempt manifest
modal run modal_attempt_features.py --max-episodes 0
```

This stage consumes the existing attempt ranges and does not perform
segmentation. Each attempt is resampled to 32 normalized timesteps. Position
trajectories use linear interpolation, wrist rotations use quaternion SLERP,
and stored features cover hand trajectories, orientations, bimanual
coordination, handedness/activity, and execution dynamics. RGB and annotation
embeddings are intentionally excluded.

Per-attempt representations are written to:

```text
/egoverse/attempt_features/<episode-id>/features.json
```

Within-task pairwise scores, greedy keep/drop decisions, and coverage curves
are written to:

```text
/egoverse/attempt_similarity/
├── summary.json
└── tasks/
    └── <task-key>.json
```

The default overall score uses trajectory/orientation/coordination/dynamics
weights of `0.45/0.25/0.20/0.10`. These weights, the physical distance scales,
activity threshold, pause duration, and redundancy threshold are configurable
from the command line. They are starting heuristics rather than learned values.

## Cluster attempts for curation and browsing

Once pairwise attempt similarities exist, generate average-linkage execution
clusters and dashboard-friendly nearest-neighbor records with:

```bash
modal run modal_attempt_clustering.py
```

This is a separate stage and does not modify attempt extraction or features.
It selects a cluster medoid, but only drops a member when at least one kept
representative meets the configured similarity threshold. Members without a
threshold-safe representative are promoted to `KEEP`.

Outputs are written to:

```text
/egoverse/attempt_clusters/
├── summary.json
├── attempt_index.json
└── tasks/
    └── <task-key>.json
```

`attempt_index.json` maps a clicked attempt ID to its task result. Each task
file contains cluster membership, medoids, keep/drop decisions, representatives,
the similarity matrix, and a similarity-ranked `similar_attempts` list for each
attempt. Every neighbor includes overall, trajectory, orientation, coordination,
and dynamics scores so a separate dashboard can explain each match.

Tune the primary threshold, neighbor-list size, and experiment thresholds with:

```bash
modal run modal_attempt_clustering.py \
  --similarity-threshold 0.90 \
  --neighbor-limit 50 \
  --experiment-thresholds 0.95,0.90,0.85,0.80
```
