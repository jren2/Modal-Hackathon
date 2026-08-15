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
