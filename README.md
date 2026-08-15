# EgoTrim · Behavioral-diversity curation for EgoVerse

EgoTrim is both a data pipeline and interactive demo for inspecting how much
EgoVerse footage can be removed while retaining the behaviors that matter. The scope of the project is only on folding clothes data but with the intent to scale up to the entire EgoVerse dataset. The current app is hosted at https://jren2--egoverse-segment-browser-web.modal.run/ with a more in depth description of the pipeline below.

Branches containing ingestion and segmentation, feature engineering, and weight tuning and clustering as split across the repo.

## Methodology

<img width="754" height="484" alt="image" src="https://github.com/user-attachments/assets/4612ce62-0086-4ebc-bd4a-141d78b10c9a" />

1. Data Ingestion
In order to ingest EgoVerse data, we utilize Modal volumes for increased parallelization and future optimizations in data processing.
2. Clip Segmentation
We extract complete task attempts from raw EgoVerse episodes while removing setup, resets, and unrelated footage. Segmentation uses three signals:

- Visual task relevance: A VLM classifies sampled RGB frames as TASK, RESET, or IRRELEVANT.
- Annotation relevance: EgoVerse annotations provide additional evidence that the current activity relates to the target task.
- Hand activity: Left/right hand velocity indicates whether active manipulation is occurring.

These signals are combined and temporally smoothed to identify contiguous task attempts.
3. Feature Engineering
Each extracted attempt is represented using features that capture how the task was physically executed:

- Hand Trajectory: Head-relative left/right hand positions over normalized time.
- Hand Orientation: Wrist orientation and rotation throughout the attempt.
- Bimanual Coordination: Inter-hand distance and relative hand positioning.
- Hand Activity: Handedness and left/right/bimanual activity patterns.
- Execution Dynamics: Duration, path length, velocity, angular velocity, and pauses.

These features form an interpretable physical fingerprint used to measure similarity between attempts and identify redundant demonstrations.
4. Weight Training
Similarity combines the engineered feature groups using configurable weights:

Trajectory: 45%
Orientation: 25%
Bimanual Coordination: 20%
Execution Dynamics: 10%

Initial weights are heuristic and can be tuned by testing which combinations best separate clearly different executions while grouping visually redundant attempts. This provides a lightweight optimization layer without requiring a learned model.
5. Clustering
We group physically similar attempts using hierarchical clustering over our weighted similarity scores. A configurable similarity threshold controls how aggressively attempts are grouped. For each cluster, we keep the medoid (most representative attempt) and mark the remaining attempts as redundant, while unique executions naturally remain as singleton clusters.

## Results
