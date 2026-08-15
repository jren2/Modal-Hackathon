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


def greedy_select(Xn, durations, budget_hours: float, log=print, per_hour: bool = True):
    """Greedily pick segments under an hour budget, maximising log det(I + K_S).

    Vendi itself is a poor greedy objective: it is not submodular, and lazy
    pruning on it is invalid because VS(S u {x}) *increases* with |S| rather than
    decreasing, so stale heap entries understate instead of overstate. Selecting
    on Vendi directly measurably underperformed random sampling.

    log det(I + K_S) *is* submodular, so greedy carries the standard (1-1/e)
    guarantee and lazy pruning is sound. By Sylvester's identity the N x N form
    equals log det(I_d + X_S^T X_S), and the matrix determinant lemma turns each
    candidate's marginal gain into

        gain(x) = log(1 + x^T A^-1 x),   A = I_d + sum_{s in S} x_s x_s^T

    which is O(d^2). Gains are divided by duration so the budget buys
    information per hour rather than per segment. The chosen set is then scored
    with Vendi, which is what we actually report.
    """
    import heapq

    import numpy as np

    n_total, d = Xn.shape
    budget_seconds = budget_hours * 3600.0
    A_inv = np.eye(d)
    chosen: list[int] = []
    spent = 0.0
    selected = np.zeros(n_total, dtype=bool)
    dur = np.asarray(durations, dtype=float)

    def gain(idx: int) -> float:
        x = Xn[idx]
        g = float(np.log1p(max(x @ A_inv @ x, 0.0)))
        if per_hour and dur[idx] > 0:
            g /= dur[idx] / 3600.0
        return g

    heap = [(-gain(i), i, 0) for i in range(n_total)]
    heapq.heapify(heap)
    evaluations = n_total

    while heap and spent < budget_seconds:
        neg, idx, stamp = heapq.heappop(heap)
        if selected[idx] or dur[idx] <= 0:
            continue
        if spent + dur[idx] > budget_seconds:
            continue
        if stamp == len(chosen):
            selected[idx] = True
            chosen.append(idx)
            # Sherman-Morrison rank-1 update of A^-1 for A <- A + x x^T
            x = Xn[idx]
            Ax = A_inv @ x
            A_inv = A_inv - np.outer(Ax, Ax) / (1.0 + x @ Ax)
            spent += dur[idx]
            if len(chosen) % 500 == 0:
                log(f"    greedy: {len(chosen)} segments, {spent / 3600:.2f}h")
        else:
            heapq.heappush(heap, (-gain(idx), idx, len(chosen)))
            evaluations += 1

    log(
        f"    greedy done: {len(chosen)} segments, {spent / 3600:.2f}h, "
        f"{evaluations} evaluations"
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


# ------------------------------------------------------------------ step 4 + 6


def _hours(meta, idx) -> float:
    return float(meta.duration.iloc[idx].sum()) / 3600.0


def _subsample_to_hours(rng, meta, idx, target_hours: float):
    """Trim a shuffled index list until it fits an hour budget.

    Subsets are compared at equal hours, not equal segment count -- otherwise a
    set of long segments wins simply by containing more footage.
    """
    order = rng.permutation(idx)
    budget = target_hours * 3600.0
    kept, spent = [], 0.0
    for i in order:
        d = float(meta.duration.iloc[i])
        if spent + d > budget:
            continue
        kept.append(int(i))
        spent += d
    return kept


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 60 * 3, memory=32768)
def compare_subsets(budget_fraction: float = 0.5, n_random_draws: int = 5) -> dict:
    """Hero comparison plus scene and operator splits, all at equal hours."""
    import numpy as np
    import pandas as pd

    Z, meta = _load()
    Xn = normalize_rows(Z)
    rng = np.random.default_rng(0)
    total_hours = float(meta.duration.sum()) / 3600.0
    budget = total_hours * budget_fraction
    out: dict = {"total_hours": round(total_hours, 3), "budget_hours": round(budget, 3),
                 "feature_dims": int(Z.shape[1])}

    # --- hero: random half vs greedy-curated half, equal hours ----------
    print(f"greedy curating to {budget:.2f}h of {total_hours:.2f}h", flush=True)
    chosen, chosen_hours = greedy_select(Xn, meta.duration.to_numpy(float), budget)
    curated = summarize(Z[chosen], chosen_hours)

    randoms = []
    for draw in range(n_random_draws):
        idx = _subsample_to_hours(rng, meta, np.arange(len(Z)), budget)
        randoms.append(summarize(Z[idx], _hours(meta, idx)))
    random_mean = {
        k: float(np.mean([r[k] for r in randoms]))
        for k in ("vendi", "nn_distance", "log_det", "hours")
    }
    out["hero"] = {
        "curated": curated,
        "random_draws": randoms,
        "random_mean": {k: round(v, 4) for k, v in random_mean.items()},
        "vendi_uplift": round(curated["vendi"] - random_mean["vendi"], 4),
        "curated_beats_random": curated["vendi"] > random_mean["vendi"],
        "coverage_pct_of_full": round(100 * curated["vendi"] / max(vendi(Z), 1e-9), 1),
    }

    # --- scene A vs scene B, same operator (hardware held constant) -----
    scene_pairs = []
    for operator, g in meta.groupby("operator"):
        counts = g.scene.value_counts()
        good = counts[counts >= 40]
        if len(good) < 2:
            continue
        a, b = good.index[0], good.index[1]
        ia = g.index[g.scene == a].to_numpy()
        ib = g.index[g.scene == b].to_numpy()
        h = min(_hours(meta, ia), _hours(meta, ib))
        if h <= 0:
            continue
        sa = _subsample_to_hours(rng, meta, ia, h)
        sb = _subsample_to_hours(rng, meta, ib, h)
        if len(sa) < 20 or len(sb) < 20:
            continue
        scene_pairs.append(
            {
                "operator": str(operator),
                "scene_a": str(a), "scene_b": str(b),
                "a": summarize(Z[sa], _hours(meta, sa)),
                "b": summarize(Z[sb], _hours(meta, sb)),
            }
        )
    for p in scene_pairs:
        p["agree"] = (
            (p["a"]["vendi"] > p["b"]["vendi"])
            == (p["a"]["nn_distance"] > p["b"]["nn_distance"])
            == (p["a"]["log_det"] > p["b"]["log_det"])
        )
    out["scene_pairs"] = scene_pairs[:10]

    # --- operator A vs B: a provenance audit, not a judgement -----------
    # This says whose recorded motion is more varied in this corpus. It is not a
    # statement about the person: it confounds task assignment, session length
    # and tracking quality, so it is only usable to spot collection anomalies.
    counts = meta.operator.value_counts()
    top = counts[counts >= 60].index[:6]
    operator_rows = []
    if len(top) >= 2:
        h = min(_hours(meta, meta.index[meta.operator == o].to_numpy()) for o in top)
        for o in top:
            idx = _subsample_to_hours(
                rng, meta, meta.index[meta.operator == o].to_numpy(), h
            )
            if len(idx) < 20:
                continue
            row = summarize(Z[idx], _hours(meta, idx))
            row["operator"] = str(o)
            row["scenes"] = int(meta.scene.iloc[idx].nunique())
            row["median_valid_frac"] = round(float(meta.valid_frac.iloc[idx].median()), 3)
            operator_rows.append(row)
    out["operators"] = {
        "equalised_hours": round(float(h), 3) if len(top) >= 2 else None,
        "rows": sorted(operator_rows, key=lambda r: -r["vendi"]),
        "caveat": (
            "Confounded by task assignment, session length and tracking quality. "
            "Use to spot collection anomalies, not to rank people."
        ),
    }

    # --- do the three measures agree? -----------------------------------
    out["hero"]["measures_agree"] = (
        curated["vendi"] > random_mean["vendi"]
        and curated["nn_distance"] > random_mean["nn_distance"]
        and curated["log_det"] > random_mean["log_det"]
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compare_subsets.json").write_text(json.dumps(out, indent=2, default=str))
    pd.DataFrame({"selected_index": chosen}).to_parquet(
        RESULTS / "curated_indices.parquet", index=False
    )
    volume.commit()
    print(json.dumps(out, indent=2, default=str)[:4000], flush=True)
    return out


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 60 * 3, memory=32768)
def curate(budget_fraction: float = 0.5) -> dict:
    """Step 6: greedy keep/drop list under an hour budget."""
    import numpy as np
    import pandas as pd

    Z, meta = _load()
    Xn = normalize_rows(Z)
    total_hours = float(meta.duration.sum()) / 3600.0
    budget = total_hours * budget_fraction

    chosen, hours = greedy_select(Xn, meta.duration.to_numpy(float), budget)
    keep = np.zeros(len(Z), dtype=bool)
    keep[chosen] = True
    full_vs = vendi(Z)
    kept_vs = vendi(Z[chosen])

    table = meta.assign(keep=keep)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_parquet(RESULTS / "curation.parquet", index=False)

    out = {
        "budget_fraction": budget_fraction,
        "total_hours": round(total_hours, 3),
        "kept_hours": round(hours, 3),
        "kept_segments": len(chosen),
        "total_segments": len(Z),
        "full_vendi": round(full_vs, 4),
        "kept_vendi": round(kept_vs, 4),
        "coverage_pct": round(100 * kept_vs / max(full_vs, 1e-9), 1),
        "headline": (
            f"{100 * hours / total_hours:.0f}% of the hours, "
            f"{100 * kept_vs / max(full_vs, 1e-9):.0f}% of the behavioural coverage"
        ),
        "kept_verb_mix": table[table.keep].verb.value_counts().head(12).to_dict(),
        "dropped_verb_mix": table[~table.keep].verb.value_counts().head(12).to_dict(),
    }
    (RESULTS / "curation.json").write_text(json.dumps(out, indent=2, default=str))
    volume.commit()
    print(json.dumps(out, indent=2, default=str))
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(sanity.remote(), indent=2))
    print(json.dumps(score_all.remote(), indent=2, default=str)[:3000])
