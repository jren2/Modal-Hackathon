"""Diversity scoring for EgoVerse-I fold-clothes — segment, featurize, cluster, score.

Implements instructions.md. Scores how varied a subset of demonstrations is by
looking at hand kinematics rather than pixels, text, or metadata labels.

Pipeline:
    1. segment    label text -> verb; merge same-verb runs into cycles
    2. featurize  wrist trajectory + fingertip configuration -> PCA
    3. cluster    HDBSCAN/k-means; name clusters by dominant verb
    4. score      Vendi over cluster histogram (composition) and within
                  clusters (execution); per-segment leave-one-out contribution

Run:
    modal run modal_diversity.py::segment
    modal run modal_diversity.py::featurize
    modal run modal_diversity.py::cluster_and_score
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

APP_NAME = "egoverse-diversity"
VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
EPISODES = MOUNT / "episodes"
META = MOUNT / "metadata"
DERIVED = MOUNT / "derived"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "zarr==3.1.5",
    "numpy==2.2.6",
    "pandas==2.3.3",
    "pyarrow==21.0.0",
    "scipy==1.15.3",
    "scikit-learn==1.6.1",
    "hdbscan==0.8.40",
)

# ---------------------------------------------------------------- step 1

# Labels are free text written by many annotators. Same motion, different string:
# "sharpen knife on whetstone" vs "... with whetstone", "smooth" vs "smoothen".
# Collapse inflection first, then map synonyms onto a canonical verb.
# Suffix stripping alone cannot decide whether to restore a silent "e"
# ("creases" -> "creas" or "crease"?). Generate candidate lemmas and accept the
# first one that is a verb we actually expect to see; fall back to the plain
# strip otherwise, so an unseen verb still normalises consistently.
_KNOWN_VERBS = {
    "adjust", "align", "apply", "arrange", "assemble", "attach", "brush",
    "carry", "clean", "close", "collect", "crease", "cut", "drape", "fill",
    "flatten", "flip", "fold", "gather", "glue", "grab", "hang", "hold",
    "insert", "iron", "lay", "lift", "lower", "move", "open", "organize",
    "pack", "peel", "pick", "place", "position", "pour", "press", "pull",
    "push", "put", "raise", "remove", "roll", "rotate", "scrub", "seal",
    "separate", "shake", "slide", "smooth", "sort", "spread", "stack",
    "straighten", "tear", "transfer", "tuck", "turn", "unfold", "unwrap",
    "wipe", "wrap",
}

_SUFFIX_RULES = [
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
]


def _candidates(word: str) -> list[str]:
    """Plausible lemmas for an inflected form, most-specific first."""
    out = [word]
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    for suffix in ("ing", "ed", "es", "s"):
        if not word.endswith(suffix) or len(word) - len(suffix) < 3:
            continue
        stem = word[: -len(suffix)]
        out.append(stem)
        out.append(stem + "e")  # creas+e, clos+e
        # flipping -> flipp -> flip
        if len(stem) > 3 and stem[-1] == stem[-2]:
            out.append(stem[:-1])
    return out

_SYNONYMS = {
    "smoothen": "smooth",
    "flatten": "smooth",
    "grab": "pick",
    "grasp": "pick",
    "take": "pick",
    "lift": "pick",
    "put": "place",
    "set": "place",
    "drop": "place",
    "straighten": "straighten",
    "unfolding": "unfold",
    "refold": "fold",
}

# Leading words that are not the action.
_LEAD_NOISE = re.compile(r"^(then|and|now|next|the|a|an|to|start|begin|continue)\s+")


def _lemma(word: str) -> str:
    """Crude but deterministic verb lemmatiser.

    A real lemmatiser (spaCy/nltk) would need a model download inside the image
    for a vocabulary of a few hundred short imperative phrases. Suffix stripping
    plus an explicit synonym table is enough here and is inspectable.
    """
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    if word in _SYNONYMS:
        return _SYNONYMS[word]
    for candidate in _candidates(word):
        mapped = _SYNONYMS.get(candidate, candidate)
        if mapped in _KNOWN_VERBS:
            return mapped
    for suffix, replacement in _SUFFIX_RULES:
        # "press" must not become "pres", "focus" must not become "focu".
        if suffix in ("s", "es") and word.endswith(("ss", "us")):
            continue
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            candidate = word[: -len(suffix)] + replacement
            return _SYNONYMS.get(candidate, candidate)
    return _SYNONYMS.get(word, word)


def label_to_verb(label: str | None) -> tuple[str, bool]:
    """Return (canonical verb, is_compound).

    39% of mecka fold labels contain a comma and describe two actions in one
    span ("pick up mask from pile, place mask on stack"). There are no sub-span
    boundaries to split on, so the first action wins and the segment is flagged
    so downstream code can exclude it rather than silently over-counting `pick`.
    """
    # 1,872 mecka segments carry a null label, which pandas surfaces as float nan.
    if not isinstance(label, str):
        return "", False
    text = label.strip().lower()
    if not text or text == "no action":
        return "", False
    compound = ("," in text) or (" and " in text)
    head = re.split(r"[,;]", text)[0].strip()
    head = _LEAD_NOISE.sub("", head)
    tokens = head.split()
    if not tokens:
        return "", compound
    verb = _lemma(tokens[0])
    # "pick up", "put down" — the particle is part of the verb but adds nothing
    # once the head verb is lemmatised, so it is dropped.
    return verb, compound


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 30)
def segment(min_duration: float = 0.5, drop_compound: bool = False) -> dict:
    """Step 1: turn labelled spans into verb cycles for episodes present on the volume."""
    import pandas as pd

    volume.reload()
    segments = pd.read_parquet(META / "segments.parquet")

    on_volume = {p.name for p in EPISODES.iterdir() if (p / "zarr.json").is_file()}
    segments = segments[segments.episode_hash.isin(on_volume)].copy()
    if segments.empty:
        raise RuntimeError(
            f"No labelled segments match the {len(on_volume)} episodes on the volume. "
            "The segments table covers mecka only — check the sync filter."
        )

    verbs = segments.label.map(label_to_verb)
    segments["verb"] = [v for v, _ in verbs]
    segments["is_compound"] = [c for _, c in verbs]
    before = len(segments)
    segments = segments[segments.verb != ""]

    segments = segments.sort_values(["episode_hash", "start_seconds"])

    # Merge consecutive same-verb spans into one cycle.
    key = (
        (segments.episode_hash != segments.episode_hash.shift())
        | (segments.verb != segments.verb.shift())
    ).cumsum()
    cycles = (
        segments.groupby(key)
        .agg(
            episode_hash=("episode_hash", "first"),
            lab=("lab", "first"),
            operator=("operator", "first"),
            scene=("scene", "first"),
            task=("task", "first"),
            verb=("verb", "first"),
            start_seconds=("start_seconds", "min"),
            end_seconds=("end_seconds", "max"),
            n_spans=("verb", "size"),
            any_compound=("is_compound", "any"),
        )
        .reset_index(drop=True)
    )
    cycles["duration"] = cycles.end_seconds - cycles.start_seconds

    merged = len(cycles)
    cycles = cycles[cycles.duration >= min_duration]
    if drop_compound:
        cycles = cycles[~cycles.any_compound]

    DERIVED.mkdir(parents=True, exist_ok=True)
    out = DERIVED / "cycles.parquet"
    cycles.to_parquet(out, index=False)
    volume.commit()

    per_episode = cycles.groupby("episode_hash").size()
    fold_per_episode = (
        cycles[cycles.verb == "fold"].groupby("episode_hash").size()
    )
    report = {
        "episodes_on_volume": len(on_volume),
        "labelled_spans": before,
        "spans_with_verb": int((segments.verb != "").sum()),
        "cycles_after_merge": merged,
        "cycles_after_filters": len(cycles),
        "compound_fraction": round(float(cycles.any_compound.mean()), 3),
        "median_cycle_seconds": round(float(cycles.duration.median()), 2),
        "cycles_per_episode_median": float(per_episode.median()),
        "fold_cycles_per_episode_median": float(fold_per_episode.median())
        if len(fold_per_episode)
        else 0.0,
        "verbs": cycles.verb.value_counts().head(25).to_dict(),
        "output": str(out),
    }
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- step 2

# MANO 21-keypoint topology: 0 = wrist, then MCP/PIP/DIP/TIP per finger.
FINGERTIPS = [4, 8, 12, 16, 20]
N_RESAMPLE = 30


def _quat_to_matrix(quaternions, w_last: bool):
    """Batch quaternion -> rotation matrix. Layout is verified empirically."""
    import numpy as np

    q = np.asarray(quaternions, dtype=float)
    if w_last:
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n[n == 0] = 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def _resample(array, n: int):
    """Linear resample along axis 0 to exactly n samples."""
    import numpy as np

    t_old = np.linspace(0.0, 1.0, len(array))
    t_new = np.linspace(0.0, 1.0, n)
    flat = array.reshape(len(array), -1)
    out = np.empty((n, flat.shape[1]), dtype=float)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_new, t_old, flat[:, j])
    return out.reshape((n,) + array.shape[1:])


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=60 * 60)
def check_quaternion_layout(n_episodes: int = 25) -> dict:
    """Decide whether obs_head_pose stores (x,y,z,w) or (w,x,y,z).

    The layout is undocumented. Under the wrong one, composing camera-frame
    keypoints with head pose produces a jittery world trajectory, so pick the
    layout that yields the smoother world-frame wrist path.
    """
    import numpy as np
    import zarr

    volume.reload()
    eps = sorted(p for p in EPISODES.iterdir() if (p / "zarr.json").is_file())
    jerk = {"w_last": [], "w_first": []}
    used = 0
    for p in eps:
        if used >= n_episodes:
            break
        try:
            g = zarr.open_group(str(p), mode="r")
            if "obs_head_pose" not in g or "right.obs_keypoints" not in g:
                continue
            hp = np.asarray(g["obs_head_pose"][:])
            kp = np.asarray(g["right.obs_keypoints"][:]).reshape(-1, 21, 3)
            m = min(len(hp), len(kp))
            hp, kp = hp[:m], kp[:m]
            ok = np.isfinite(hp).all(1) & np.isfinite(kp).all((1, 2))
            hp, kp = hp[ok][:600], kp[ok][:600]
            if len(hp) < 50:
                continue
            for name, w_last in (("w_last", True), ("w_first", False)):
                R = _quat_to_matrix(hp[:, 3:7], w_last)
                wrist_world = np.einsum("tij,tj->ti", R, kp[:, 0]) + hp[:, :3]
                d2 = np.diff(wrist_world, n=2, axis=0)
                jerk[name].append(float(np.median(np.linalg.norm(d2, axis=-1))))
            used += 1
        except Exception:
            continue

    med = {k: float(np.median(v)) if v else float("inf") for k, v in jerk.items()}
    choice = min(med, key=med.get)
    result = {
        "episodes_used": used,
        "median_world_wrist_jerk": med,
        "chosen_layout": choice,
        "note": "lower jerk = correct layout",
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    timeout=60 * 60 * 4,
    cpu=4,
    memory=16384,
)
def featurize(
    w_last: bool = True,
    min_valid_frac: float = 0.7,
    n_resample: int = N_RESAMPLE,
    pca_dims: int = 30,
) -> dict:
    """Step 2: per-cycle wrist trajectory + fingertip configuration -> PCA."""
    import numpy as np
    import pandas as pd
    import zarr
    from sklearn.decomposition import PCA

    volume.reload()
    cycles = pd.read_parquet(DERIVED / "cycles.parquet")

    rows, feats = [], []
    skipped = {"missing_episode": 0, "missing_array": 0, "too_short": 0, "low_valid": 0}

    for episode_hash, group in cycles.groupby("episode_hash"):
        path = EPISODES / episode_hash
        if not (path / "zarr.json").is_file():
            skipped["missing_episode"] += len(group)
            continue
        try:
            g = zarr.open_group(str(path), mode="r")
            needed = ["left.obs_keypoints", "right.obs_keypoints", "obs_head_pose"]
            if any(k not in g for k in needed):
                skipped["missing_array"] += len(group)
                continue
            fps = float(dict(g.attrs).get("fps", 30))
            left = np.asarray(g["left.obs_keypoints"][:]).reshape(-1, 21, 3)
            right = np.asarray(g["right.obs_keypoints"][:]).reshape(-1, 21, 3)
            head = np.asarray(g["obs_head_pose"][:])
        except Exception:
            skipped["missing_array"] += len(group)
            continue

        n = min(len(left), len(right), len(head))
        left, right, head = left[:n], right[:n], head[:n]

        for row in group.itertuples():
            a = int(round(row.start_seconds * fps))
            b = int(round(row.end_seconds * fps))
            a, b = max(0, a), min(n, b)
            if b - a < 5:
                skipped["too_short"] += 1
                continue
            hands = [left[a:b], right[a:b]]
            hp = head[a:b]
            valid = (
                np.isfinite(hp).all(1)
                & np.isfinite(hands[0]).all((1, 2))
                & np.isfinite(hands[1]).all((1, 2))
            )
            frac = float(valid.mean())
            if frac < min_valid_frac or valid.sum() < 5:
                skipped["low_valid"] += 1
                continue

            hp_v = hp[valid]
            R = _quat_to_matrix(hp_v[:, 3:7], w_last)
            R0t = R[0].T
            p0 = hp_v[0, :3]

            wrist_parts, tip_parts = [], []
            for hand in hands:
                h = hand[valid]
                # camera -> world -> head frame at t=0
                world = np.einsum("tij,tkj->tki", R, h) + hp_v[:, None, :3]
                local = np.einsum("ij,tkj->tki", R0t, world - p0)
                wrist = local[:, 0, :]
                wrist_parts.append(_resample(wrist - wrist[0], n_resample))
                # fingertips relative to their own wrist: viewpoint- and
                # placement-invariant, so this is pure hand configuration
                tips = h[:, FINGERTIPS, :] - h[:, 0:1, :]
                tip_parts.append(_resample(tips, n_resample))

            vector = np.concatenate(
                [np.concatenate(wrist_parts, -1).ravel(),
                 np.concatenate(tip_parts, -1).ravel(),
                 [row.duration]]
            )
            if not np.isfinite(vector).all():
                skipped["low_valid"] += 1
                continue
            feats.append(vector)
            rows.append(
                {
                    "episode_hash": episode_hash,
                    "verb": row.verb,
                    "operator": row.operator,
                    "scene": row.scene,
                    "task": row.task,
                    "start_seconds": row.start_seconds,
                    "end_seconds": row.end_seconds,
                    "duration": row.duration,
                    "any_compound": bool(row.any_compound),
                    "valid_frac": frac,
                }
            )

    if not feats:
        raise RuntimeError(f"No usable cycles. skipped={skipped}")

    X = np.asarray(feats)
    meta = pd.DataFrame(rows)

    # Standardise the three blocks so a 900-dim configuration block cannot
    # drown out the 180-dim trajectory block or the single duration scalar.
    n_wrist = n_resample * 3 * 2
    n_tips = n_resample * len(FINGERTIPS) * 3 * 2
    blocks = [(0, n_wrist), (n_wrist, n_wrist + n_tips), (n_wrist + n_tips, X.shape[1])]
    Xs = X.copy()
    for lo, hi in blocks:
        block = Xs[:, lo:hi]
        scale = block.std() or 1.0
        Xs[:, lo:hi] = (block - block.mean(0)) / scale

    dims = min(pca_dims, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=dims, random_state=0)
    Z = pca.fit_transform(Xs)

    DERIVED.mkdir(parents=True, exist_ok=True)
    np.save(DERIVED / "features.npy", Z)
    meta.to_parquet(DERIVED / "features_meta.parquet", index=False)
    volume.commit()

    report = {
        "cycles_in": len(cycles),
        "cycles_featurised": len(meta),
        "skipped": skipped,
        "raw_dims": int(X.shape[1]),
        "pca_dims": dims,
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 3),
        "median_valid_frac": round(float(meta.valid_frac.median()), 3),
        "verbs": meta.verb.value_counts().head(15).to_dict(),
    }
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(min_duration: float = 0.5, drop_compound: bool = False):
    print(json.dumps(segment.remote(min_duration, drop_compound), indent=2))
