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
3. Score

See explanation below. Compute global Vendi, per-verb Vendi, per-segment contribution, per-video internal + contribution.

4. Compare subsets
Hero: random 50% vs. greedy-curated 50% at equal hours — your metric should pick the better half
Scene A vs. scene B, same partner (hardware held constant)
Partner A vs. B as a provenance audit, framed carefully
5. Dashboard

Per-verb Vendi bars · subset comparison panel · ranked video table (contribution, internal diversity, tracking validity) · scatter of internal vs. contribution · top-5 vs. bottom-5 video strip

6. Stretch

Greedy selection under an hour budget → keep/drop list. Track 1 for free.

How the scoring works
The core function

Everything is one function applied to different sets.

Step 1 — kernel. Take your N segment vectors, row-normalize to unit length, then:

K = X @ X.T        # cosine similarity, N×N, diagonal = 1
K = K / N          # now trace = 1

Step 2 — eigenvalues.

λ = eigvalsh(K)
λ = λ[λ > 1e-12]

Because trace = 1, the eigenvalues sum to 1 and behave like a probability distribution.

Step 3 — Vendi.

VS = exp(-Σ λᵢ log λᵢ)

Why this works: the eigenvalues describe how the data's "mass" spreads across independent directions. If everything is identical, one eigenvalue is 1 and the rest are 0 — entropy 0, VS = 1. If everything is mutually dissimilar, mass spreads evenly across N eigenvalues — entropy log N, VS = N.

So VS reads as the effective number of distinct behaviors. 800 segments scoring 34 means: behaviorally, you have about 34 things, repeated.

Speed: with d=30 features, use the dual form — eigenvalues of X.T @ X / N (30×30) match those of the N×N kernel. Instant regardless of N, which makes leave-one-out affordable.

Applied at four granularities

Global — Vendi(all segments). One number for the whole fold-clothes corpus.

Per verb — group segments by annotation, Vendi within each group. The grouping comes from labels, the measurement from kinematics, so there's no circularity. Output is a vector: {fold: 18, smooth: 4, pick: 9}. Only score verbs with enough N — below ~50 segments the eigenvalue estimate is noisy.

Per segment (contribution) —

contribution(x) = Vendi(group) − Vendi(group \ x)

Computed within the segment's verb group. So a fold competes against other folds, and "distinctive" means "unusual way of folding" rather than "this is a fold." Rank and report as percentile.

Per video — two different numbers:

internal(v) = Vendi(v's segments) — how varied one session is
contribution(v) = Vendi(all) − Vendi(all \ v's segments) — leave the whole video out, not each segment separately, so five near-identical folds don't get counted five times
Comparing subsets

Same function, different input: Vendi(A) vs Vendi(B), both globally and per-verb. Normalize by hours, not segment count.

Also compute NN-distance and log det(K + I) over the same kernel — they're free. If all three rank A and B the same way, say so. That sentence is worth more than any single number.

Sanity checks before trusting any of it
Duplicate the dataset 2× → VS should barely move, not double
Random Gaussian vectors → VS ≈ N
Identical vectors → VS = 1