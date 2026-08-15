Diversity scoring — EgoVerse-I fold-clothes, all actions

1. Segment (already completed)
Lemmatize annotation text → verb
Merge consecutive same-verb windows → one cycle
Drop cycles < 0.5s
Merge duplicates (smooth/smoothen, grab/pick)
Keep all verbs present: pick, fold, adjust, smooth, straighten, flip, spread, unfold, etc.
Drop segments with tracking_valid_frac < 0.7
Sanity: ~3 fold cycles per episode
2. Featurize

Per segment:

Resample to 30 frames
Project into head frame at t=0, subtract wrist position at t=0
Two signals:
wrist trajectory — 30 × 3 × 2 hands = 180 dims
fingertip config relative to own wrist — joints 4, 8, 12, 16, 20 → 900 dims
Append duration as scalar
PCA → ~30 dims
Sanity: distance matrix on 20 segments — nearest neighbors should look alike
3. Cluster
HDBSCAN or k-means over all segments
Name clusters by dominant verb ("cluster 3 = 78% fold")
Check: do clusters recover the verb labels, or cut across them? Either answer is interesting — if a single verb splits into three clusters, you've found styles the annotation can't see.
4. Score

Per subset:

Composition — Vendi over the cluster histogram. Which actions appear, in what mix.
Execution — Vendi within each cluster, weighted mean. How variably each action is performed.
Cross-check with NN-distance, log-det

Per segment (headline):

contribution(i) = Vendi(all) − Vendi(all \ i)
Report as percentile
Fallback: mean distance to k nearest neighbors

Per episode: aggregate its segments' contributions → rank whole videos too

5. Dashboard

Cluster composition bars · composition vs. execution scores · ranked segment table with distinctiveness percentile · top-vs-bottom video strip

6. Stretch — curation

Greedy select segments maximizing Vendi under an hour budget → keep/drop list. "40% of the data, 95% of the behavioral coverage." Track 1 for free.