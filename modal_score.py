"""Vendi diversity scoring for EgoVerse-I fold-clothes kinematics.

Implements instructions.md steps 3, 4 and 6. One kernel function applied at four
granularities, then subset comparisons and greedy curation.

The Vendi score reads as *the effective number of distinct behaviours*: identical
segments score 1, mutually dissimilar segments score N. Everything below is that
one function applied to different row sets.

    modal run modal_score.py::sanity          # verify the estimator first
    modal run modal_score.py::score_all
    modal run modal_score.py::compare_subsets
    modal run modal_score.py::curate --budget-fraction 0.5
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "egoverse-score"
VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
DERIVED = MOUNT / "derived"
RESULTS = DERIVED / "results"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.2.6", "pandas==2.3.3", "pyarrow==21.0.0", "scipy==1.15.3"
)

EPS = 1e-12
# Below this many segments the eigenvalue estimate is too noisy to report.
MIN_GROUP = 50


# ------------------------------------------------------------------ core math
#
# Pure numpy, no Modal — so the estimator can be unit-tested locally against the
# sanity checks in instructions.md before any of it touches real data.


def normalize_rows(X):
    """Row-normalise to unit length so the kernel is cosine similarity."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    return X / norms


def gram(Xn):
    """d x d Gram matrix of already-normalised rows.

    The N x N kernel K = Xn @ Xn.T / N and the d x d matrix Xn.T @ Xn / N share
    their non-zero eigenvalues, so with d=30 every quantity below is cheap
    regardless of N. That is what makes leave-one-out affordable.
    """
    return Xn.T @ Xn


def eigenvalues_from_gram(G, n: int):
    """Non-zero eigenvalues of the N x N kernel, via the d x d dual form."""
    import numpy as np

    if n <= 0:
        return np.zeros(0)
    lam = np.linalg.eigvalsh(G / n)
    return lam[lam > EPS]


def _vendi_lambda(lam) -> float:
    """Shannon entropy of the eigenvalue spectrum, exponentiated."""
    import numpy as np

    lam = lam[lam > EPS]
    if lam.size == 0:
        return 0.0
    return float(np.exp(-(lam * np.log(lam)).sum()))


def vendi_from_gram(G, n: int) -> float:
    return _vendi_lambda(eigenvalues_from_gram(G, n))


def vendi(X) -> float:
    """Effective number of distinct behaviours among the rows of X."""
    Xn = normalize_rows(X)
    if len(Xn) == 0:
        return 0.0
    return vendi_from_gram(gram(Xn), len(Xn))


def vendi_leave_out(Xn, G, index_or_indices) -> float:
    """Vendi of the set with given rows removed, without rebuilding the Gram.

    G is the (unnormalised) Gram of Xn, so removing rows is a rank-k downdate:
    G' = G - R.T @ R. O(d^2) per removal instead of O(N d^2).
    """
    import numpy as np

    idx = np.atleast_1d(index_or_indices)
    R = Xn[idx]
    n = len(Xn) - len(idx)
    if n <= 0:
        return 0.0
    G2 = G - R.T @ R
    return _vendi_lambda(np.linalg.eigvalsh(G2 / n))


def nn_distance(Xn) -> float:
    """Mean cosine distance to nearest neighbour. A cheap independent read."""
    import numpy as np

    n = len(Xn)
    if n < 2:
        return 0.0
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    return float(np.mean(1.0 - sim.max(axis=1)))


def log_det(Xn) -> float:
    """log det(K + I) over the same kernel; zero eigenvalues contribute nothing."""
    import numpy as np

    n = len(Xn)
    if n == 0:
        return 0.0
    lam = np.linalg.eigvalsh(gram(Xn) / n)
    lam = lam[lam > EPS]
    return float(np.sum(np.log1p(lam)))


def summarize(X, hours: float | None = None) -> dict:
    """All three diversity reads for one set of segments."""
    Xn = normalize_rows(X)
    out = {
        "n": int(len(Xn)),
        "vendi": vendi(X),
        "nn_distance": nn_distance(Xn),
        "log_det": log_det(Xn),
    }
    if hours:
        out["hours"] = round(float(hours), 3)
        out["vendi_per_hour"] = round(out["vendi"] / hours, 4) if hours > 0 else 0.0
    return out


def greedy_select(Xn, durations, budget_hours: float, log=print):
    """Greedily pick segments maximising Vendi until the hour budget is spent.

    Lazy (CELF) evaluation: the marginal gain of a candidate never increases as
    the selected set grows, so a stale gain that still beats every other
    candidate's upper bound is safe to accept. Vendi is not provably submodular,
    so this is a heuristic rather than a guaranteed (1-1/e) approximation --
    a full re-evaluation pass costs O(N d^3) per step and is not affordable.
    """
    import heapq

    import numpy as np

    n_total, d = Xn.shape
    budget_seconds = budget_hours * 3600.0
    G = np.zeros((d, d))
    chosen: list[int] = []
    spent = 0.0
    selected = np.zeros(n_total, dtype=bool)

    def score_with(idx: int) -> float:
        return _vendi_lambda(
            np.linalg.eigvalsh((G + np.outer(Xn[idx], Xn[idx])) / (len(chosen) + 1))
        )

    current = 0.0
    heap = [(-score_with(i), i, 0) for i in range(n_total)]
    heapq.heapify(heap)

    evaluations = n_total
    while heap and spent < budget_seconds:
        neg_gain, idx, stamp = heapq.heappop(heap)
        if selected[idx]:
            continue
        if durations[idx] <= 0 or spent + durations[idx] > budget_seconds:
            continue
        if stamp == len(chosen):
            selected[idx] = True
            chosen.append(idx)
            G = G + np.outer(Xn[idx], Xn[idx])
            current = -neg_gain
            spent += durations[idx]
            if len(chosen) % 500 == 0:
                log(f"    greedy: {len(chosen)} segments, {spent / 3600:.2f}h")
        else:
            heapq.heappush(heap, (-(score_with(idx)), idx, len(chosen)))
            evaluations += 1

    log(
        f"    greedy done: {len(chosen)} segments, {spent / 3600:.2f}h, "
        f"{evaluations} evaluations, VS={current:.2f}"
    )
    return chosen, spent / 3600.0


# ------------------------------------------------------------------ step 3


def _load(volume_reload=True):
    import numpy as np
    import pandas as pd

    if volume_reload:
        volume.reload()
    Z = np.load(DERIVED / "features.npy")
    meta = pd.read_parquet(DERIVED / "features_meta.parquet")
    if len(Z) != len(meta):
        raise RuntimeError(f"features {len(Z)} != meta {len(meta)}")
    return Z, meta


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 30)
def sanity() -> dict:
    """Run the spec's sanity checks against the real feature matrix."""
    import numpy as np

    Z, meta = _load()
    Xn = normalize_rows(Z)
    rng = np.random.default_rng(0)

    identical = np.repeat(Z[:1], 200, axis=0)
    gaussian = rng.normal(size=(200, Z.shape[1]))
    subset = Z[rng.choice(len(Z), min(400, len(Z)), replace=False)]

    out = {
        "feature_dims": int(Z.shape[1]),
        "rank_cap": int(min(Z.shape[1], len(Z))),
        "identical_vectors_vs": round(vendi(identical), 4),
        "random_gaussian_vs": round(vendi(gaussian), 4),
        "real_subset_vs": round(vendi(subset), 4),
        "real_subset_duplicated_vs": round(vendi(np.concatenate([subset, subset])), 4),
        "global_vs": round(vendi(Z), 4),
        "nn_distance": round(nn_distance(Xn), 4),
    }
    out["duplication_stable"] = (
        abs(out["real_subset_vs"] - out["real_subset_duplicated_vs"]) < 0.05
        * max(out["real_subset_vs"], 1)
    )
    # A cosine kernel on d-dimensional features has rank <= d, so the Vendi
    # score cannot exceed d no matter how many distinct segments exist.
    out["vs_capped_by_dims"] = out["global_vs"] > 0.9 * Z.shape[1]
    print(json.dumps(out, indent=2))
    return out


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 60 * 2, memory=16384)
def score_all(min_group: int = MIN_GROUP) -> dict:
    """Global, per-verb, per-segment and per-video Vendi."""
    import numpy as np
    import pandas as pd

    Z, meta = _load()
    Xn = normalize_rows(Z)
    hours_total = float(meta.duration.sum()) / 3600.0

    results: dict = {
        "feature_dims": int(Z.shape[1]),
        "rank_cap": int(Z.shape[1]),
        "global": summarize(Z, hours_total),
    }

    # --- per verb -------------------------------------------------------
    per_verb = []
    for verb, g in meta.groupby("verb"):
        idx = g.index.to_numpy()
        entry = summarize(Z[idx], float(g.duration.sum()) / 3600.0)
        entry["verb"] = verb
        entry["reliable"] = len(idx) >= min_group
        per_verb.append(entry)
    per_verb.sort(key=lambda r: -r["n"])
    results["per_verb"] = per_verb
    results["min_group"] = min_group

    # --- per segment: contribution within its own verb group ------------
    # A fold competes against other folds, so "distinctive" means an unusual
    # way of folding rather than simply "this is a fold".
    contribution = np.full(len(Z), np.nan)
    percentile = np.full(len(Z), np.nan)
    for verb, g in meta.groupby("verb"):
        idx = g.index.to_numpy()
        if len(idx) < 3:
            continue
        Xg = Xn[idx]
        Gg = gram(Xg)
        base = vendi_from_gram(Gg, len(idx))
        vals = np.array(
            [base - vendi_leave_out(Xg, Gg, i) for i in range(len(idx))]
        )
        contribution[idx] = vals
        order = vals.argsort().argsort()
        percentile[idx] = 100.0 * order / max(len(idx) - 1, 1)
    meta = meta.assign(contribution=contribution, contribution_pct=percentile)

    # --- per video ------------------------------------------------------
    G_all = gram(Xn)
    vs_all = vendi_from_gram(G_all, len(Xn))
    videos = []
    for episode_hash, g in meta.groupby("episode_hash"):
        idx = g.index.to_numpy()
        internal = vendi(Z[idx]) if len(idx) > 1 else 1.0
        # Leave the whole video out, so five near-identical folds inside one
        # session are not credited five times over.
        contrib = vs_all - vendi_leave_out(Xn, G_all, idx)
        videos.append(
            {
                "episode_hash": episode_hash,
                "operator": g.operator.iloc[0],
                "scene": g.scene.iloc[0],
                "task": g.task.iloc[0],
                "n_segments": int(len(idx)),
                "hours": round(float(g.duration.sum()) / 3600.0, 4),
                "internal_vendi": round(float(internal), 4),
                "contribution": round(float(contrib), 6),
                "median_valid_frac": round(float(g.valid_frac.median()), 3),
                "verbs": ",".join(sorted(set(g.verb))),
            }
        )
    videos.sort(key=lambda r: -r["contribution"])
    results["n_videos"] = len(videos)

    RESULTS.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(RESULTS / "segments_scored.parquet", index=False)
    pd.DataFrame(videos).to_parquet(RESULTS / "videos_scored.parquet", index=False)
    (RESULTS / "score_all.json").write_text(json.dumps(results, indent=2, default=str))
    volume.commit()

    results["top_videos"] = videos[:5]
    results["bottom_videos"] = videos[-5:]
    print(json.dumps(results, indent=2, default=str))
    return results
