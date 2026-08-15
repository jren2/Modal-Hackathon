
## Dashboard

The demo includes:

- an eight-metric curation overview;
- an adjustable training-hours budget comparing diversity-based and random
  selection;
- before/after action-composition bars;
- a searchable, filterable, row-selectable segment explorer;
- clip detail and five cross-video behavioral neighbors; and
- a verb-colored episode timeline with retained segments highlighted.

### Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The included `mock_results.json` is loaded by default. Mock clip paths point to
the future `clips/` output location; until those files exist, the dashboard
shows polished media placeholders instead of raising playback errors.

### Swap in pipeline output

All external data access is isolated in `load_results()` in `app.py`. Either
replace the body of that function or point the existing JSON loader at a new
artifact without changing the rest of the UI:

```powershell
$env:EGOTRIM_RESULTS_PATH = "C:\path\to\results.json"
streamlit run app.py
```

Paths can be absolute or relative to the repository root. The loader tolerates
missing summaries, curve points, action rows, optional segment metadata,
neighbor arrays, and clip files. It derives sensible counts where possible and
surfaces an empty-state message where it cannot.

### JSON contract

```json
{
  "summary": {
    "total_videos": 100,
    "total_segments": 850,
    "original_hours": 12.4,
    "selected_hours": 5.0,
    "coverage_retained": 94,
    "composition_diversity": 86,
    "execution_diversity": 91
  },
  "coverage_curve": [
    {
      "retained_hours": 5.0,
      "retained_percent": 40.3,
      "diversity_coverage": 94,
      "random_coverage": 72
    }
  ],
  "actions": [
    {"verb": "fold", "before": 180, "after": 85}
  ],
  "segments": [
    {
      "id": "segment_001",
      "episode_id": "episode_01",
      "verb": "fold",
      "start": 12.5,
      "end": 14.8,
      "clip_path": "clips/segment_001.mp4",
      "cluster": "fold-style-2",
      "distinctiveness_percentile": 91,
      "keep": true,
      "neighbors": [
        {"segment_id": "segment_105", "similarity": 0.94}
      ]
    }
  ]
}
```

`coverage_curve`, `actions`, `segments`, and individual optional fields may be
omitted. Neighbor relations are resolved against segment IDs and cross-video
matches only; `video_id` is optional and falls back to `episode_id`.

## EgoVerse data utilities

The repository also contains the existing ingestion, segmentation, bounded
subset download, and feature-extraction utilities. Those are independent of
the mock dashboard.

### Download EgoVerse into Modal

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

## EgoTrim bounded subset utility

EgoTrim curates complete EgoVerse episodes under a fixed data budget. The first
utility safely obtains a small real-data subset for feature extraction and the
demo.

## Download a bounded EgoVerse subset

The downloader uses the official GaTech-RL2/EgoVerse SQL and storage APIs. It
queries current episode metadata first, selects a stable seeded sample, prints
the episode IDs/labs/scenes/demonstrators, and only then asks the official
`S3EpisodeResolver` to download those exact Zarrs.

Prerequisites:

1. Clone and install [GaTech-RL2/EgoVerse](https://github.com/GaTech-RL2/EgoVerse).
2. Configure the official read-only AWS/R2 environment with EgoVerse's
   `egomimic/utils/aws/setup_secret.sh`. Never commit the resulting environment
   file.
3. Ensure `s5cmd` is available on `PATH`.

Preview an exact task without writing anything:

```powershell
python scripts/download_egoverse_subset.py `
  --task fold_clothes `
  --max-episodes 10 `
  --seed 42 `
  --dry-run `
  --egoverse-repo C:\path\to\EgoVerse
```

Remove `--dry-run` to download to `data/egoverse_subset/`. Use the task string
exactly as it appears in the EgoVerse SQL table. An all-task (`*`/`all`) or
unlimited (`--max-episodes 0`) request is rejected unless
`--confirm-complete-dataset` is explicitly supplied.

## Put the subset in the Modal Volume

After the bounded local download completes, upload that folder to the existing
`egotrim-data` Volume:

```powershell
modal volume put egotrim-data data/egoverse_subset /egoverse_subset
modal volume ls egotrim-data /egoverse_subset
```

The resulting path inside a Modal function is
`<volume-mount>/egoverse_subset`. Mount the volume at `/data` to access it as
`/data/egoverse_subset`.

## Tests

```powershell
python -m unittest discover -s tests -v
```
