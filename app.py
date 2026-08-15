"""EgoTrim — a mock-data Streamlit dashboard for EgoVerse curation results.

The eventual pipeline integration point is intentionally limited to
``load_results``. Everything below that function consumes its normalized return
value, so swapping the JSON file for a pipeline artifact does not affect the UI.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_left
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = APP_DIR / "mock_results.json"

BACKGROUND = "#070B14"
PANEL = "#0D1422"
PANEL_LIGHT = "#111B2C"
TEXT = "#F4F7FB"
MUTED = "#8C9BB3"
GRID = "rgba(140, 155, 179, 0.12)"
CYAN = "#51E5D4"
PURPLE = "#A98BFF"
GOLD = "#FFCF66"
RED = "#FF718B"

VERB_COLORS = {
    "fold": "#51E5D4",
    "pick": "#A98BFF",
    "smooth": "#FFCF66",
    "adjust": "#FF718B",
    "straighten": "#63A9FF",
    "flip": "#F195D2",
    "spread": "#85D66D",
    "unfold": "#FF9E64",
}


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result else fallback  # NaN-safe
    except (TypeError, ValueError):
        return fallback


def _integer(value: Any, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _boolean(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "keep", "kept"}:
            return True
        if normalized in {"false", "0", "no", "drop", "dropped"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


@st.cache_data(show_spinner=False)
def load_results(results_path: str | Path | None = None) -> dict[str, Any]:
    """Load and normalize one curation-results artifact.

    Replace the body of this function when the real pipeline is ready. The UI
    deliberately performs no file, API, database, or Modal reads elsewhere.
    Missing optional fields receive safe defaults; a missing/unreadable artifact
    returns an empty dashboard plus a human-readable ``source_error``.
    """

    configured_path = results_path or os.getenv(
        "EGOTRIM_RESULTS_PATH", str(DEFAULT_RESULTS_PATH)
    )
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("the top-level JSON value must be an object")
        source_error = ""
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raw = {}
        source_error = f"Could not load {path.name}: {exc}"

    raw_segments = _list(raw.get("segments"))
    segments: list[dict[str, Any]] = []
    for index, item in enumerate(raw_segments, start=1):
        segment = _mapping(item)
        start = max(0.0, _number(segment.get("start")))
        end = max(start, _number(segment.get("end"), start))
        segment_id = str(segment.get("id") or f"segment_{index:03d}")
        episode_id = str(segment.get("episode_id") or "unknown_episode")
        neighbors: list[dict[str, Any]] = []
        for neighbor in _list(segment.get("neighbors")):
            neighbor_data = _mapping(neighbor)
            neighbor_id = neighbor_data.get("segment_id") or neighbor_data.get("id")
            if neighbor_id:
                neighbors.append(
                    {
                        "segment_id": str(neighbor_id),
                        "similarity": min(
                            1.0, max(0.0, _number(neighbor_data.get("similarity")))
                        ),
                        "explanation": str(neighbor_data.get("explanation") or ""),
                    }
                )

        segments.append(
            {
                "id": segment_id,
                "episode_id": episode_id,
                "video_id": str(segment.get("video_id") or episode_id),
                "verb": str(segment.get("verb") or "unlabeled").lower(),
                "start": start,
                "end": end,
                "clip_path": str(segment.get("clip_path") or ""),
                "cluster": str(segment.get("cluster") or "unclustered"),
                "distinctiveness_percentile": min(
                    100.0,
                    max(0.0, _number(segment.get("distinctiveness_percentile"))),
                ),
                "keep": _boolean(segment.get("keep")),
                "neighbors": neighbors,
            }
        )

    summary_raw = _mapping(raw.get("summary"))
    episode_count = len({segment["episode_id"] for segment in segments})
    observed_hours = sum(segment["end"] - segment["start"] for segment in segments) / 3600
    observed_selected_hours = sum(
        segment["end"] - segment["start"] for segment in segments if segment["keep"]
    ) / 3600
    original_hours = _number(summary_raw.get("original_hours"), observed_hours)
    selected_hours = _number(summary_raw.get("selected_hours"), observed_selected_hours)
    reduction = (
        max(0.0, min(100.0, (1 - selected_hours / original_hours) * 100))
        if original_hours > 0
        else 0.0
    )
    summary = {
        "total_videos": _integer(summary_raw.get("total_videos"), episode_count),
        "total_segments": _integer(summary_raw.get("total_segments"), len(segments)),
        "original_hours": original_hours,
        "selected_hours": selected_hours,
        "data_reduction_percent": _number(
            summary_raw.get("data_reduction_percent"), reduction
        ),
        "coverage_retained": _number(summary_raw.get("coverage_retained")),
        "composition_diversity": _number(
            summary_raw.get("composition_diversity")
        ),
        "execution_diversity": _number(summary_raw.get("execution_diversity")),
    }

    coverage_curve: list[dict[str, float]] = []
    for point in _list(raw.get("coverage_curve")):
        value = _mapping(point)
        retained_hours = _number(value.get("retained_hours"))
        retained_percent = _number(value.get("retained_percent"))
        if not retained_hours and retained_percent and original_hours > 0:
            retained_hours = retained_percent / 100 * original_hours
        if not retained_percent and original_hours > 0:
            retained_percent = retained_hours / original_hours * 100
        coverage_curve.append(
            {
                "retained_hours": retained_hours,
                "retained_percent": max(0.0, min(100.0, retained_percent)),
                "diversity_coverage": max(
                    0.0, min(100.0, _number(value.get("diversity_coverage")))
                ),
                "random_coverage": max(
                    0.0, min(100.0, _number(value.get("random_coverage")))
                ),
            }
        )
    coverage_curve.sort(key=lambda point: point["retained_hours"])

    actions: list[dict[str, Any]] = []
    for item in _list(raw.get("actions")):
        action = _mapping(item)
        verb = str(action.get("verb") or action.get("action") or "unlabeled").lower()
        actions.append(
            {
                "verb": verb,
                "before": max(
                    0.0, _number(action.get("before"), _number(action.get("before_count")))
                ),
                "after": max(
                    0.0, _number(action.get("after"), _number(action.get("after_count")))
                ),
            }
        )
    if not actions and segments:
        for verb in sorted({segment["verb"] for segment in segments}):
            actions.append(
                {
                    "verb": verb,
                    "before": sum(segment["verb"] == verb for segment in segments),
                    "after": sum(
                        segment["verb"] == verb and segment["keep"]
                        for segment in segments
                    ),
                }
            )

    return {
        "summary": summary,
        "coverage_curve": coverage_curve,
        "actions": actions,
        "segments": segments,
        "source_path": str(path),
        "source_error": source_error,
    }


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: dark; }}
        .stApp {{
            background:
                radial-gradient(circle at 82% 3%, rgba(81,229,212,.09), transparent 24rem),
                radial-gradient(circle at 9% 25%, rgba(169,139,255,.07), transparent 28rem),
                {BACKGROUND};
            color: {TEXT};
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{ right: 1rem; }}
        .block-container {{ max-width: 1450px; padding: 2.3rem 2.5rem 5rem; }}
        h1, h2, h3 {{ letter-spacing: -0.035em; color: {TEXT}; }}
        h2 {{ margin-top: .3rem; }}
        p, label, [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
        .hero {{
            position: relative; overflow: hidden; padding: 2rem 2.1rem 1.9rem;
            border: 1px solid rgba(255,255,255,.09); border-radius: 24px;
            background: linear-gradient(120deg, rgba(17,27,44,.94), rgba(10,17,29,.82));
            box-shadow: 0 24px 80px rgba(0,0,0,.28); margin-bottom: 1.7rem;
        }}
        .hero::after {{
            content: ''; position: absolute; width: 260px; height: 260px;
            border-radius: 50%; right: -70px; top: -120px;
            background: {CYAN}; filter: blur(100px); opacity: .16;
        }}
        .eyebrow {{
            color: {CYAN}; font-size: .74rem; font-weight: 800; letter-spacing: .16em;
            text-transform: uppercase; margin-bottom: .65rem;
        }}
        .hero h1 {{ font-size: clamp(2.2rem, 5vw, 4rem); line-height: .98; margin: 0 0 .8rem; }}
        .hero p {{ max-width: 700px; font-size: 1.03rem; line-height: 1.65; margin: 0; }}
        .live-pill {{
            position: absolute; right: 2rem; bottom: 2rem; z-index: 2;
            padding: .48rem .75rem; border-radius: 999px; color: {CYAN};
            background: rgba(81,229,212,.08); border: 1px solid rgba(81,229,212,.2);
            font-size: .72rem; font-weight: 800; letter-spacing: .08em;
        }}
        .section-heading {{ display: flex; align-items: end; justify-content: space-between; margin: 2.4rem 0 1rem; }}
        .section-heading h2 {{ margin: 0; font-size: 1.6rem; }}
        .section-heading span {{ color: {MUTED}; font-size: .84rem; }}
        .metric-card {{
            min-height: 118px; padding: 1rem 1.05rem; border-radius: 18px;
            background: linear-gradient(145deg, rgba(17,27,44,.96), rgba(12,19,32,.94));
            border: 1px solid rgba(255,255,255,.075); box-shadow: 0 12px 38px rgba(0,0,0,.17);
        }}
        .metric-label {{ color: {MUTED}; font-size: .76rem; font-weight: 700; letter-spacing: .03em; min-height: 2.2em; }}
        .metric-value {{ color: {TEXT}; font-size: 1.75rem; font-weight: 760; letter-spacing: -.04em; margin-top: .35rem; }}
        .metric-foot {{ color: {CYAN}; font-size: .72rem; margin-top: .28rem; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(13,20,34,.72); border-color: rgba(255,255,255,.075) !important;
            border-radius: 20px; box-shadow: 0 18px 48px rgba(0,0,0,.15);
        }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div,
        [data-testid="stTextInputRootElement"] {{
            background: {PANEL_LIGHT} !important; border-color: rgba(255,255,255,.09) !important;
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid rgba(255,255,255,.08); border-radius: 16px; overflow: hidden;
        }}
        .badge {{
            display: inline-block; padding: .3rem .58rem; margin: 0 .35rem .35rem 0;
            border-radius: 999px; background: rgba(169,139,255,.1); color: #CFBEFF;
            border: 1px solid rgba(169,139,255,.22); font-size: .75rem; font-weight: 700;
        }}
        .badge.keep {{ background: rgba(81,229,212,.09); color: {CYAN}; border-color: rgba(81,229,212,.22); }}
        .clip-placeholder {{
            min-height: 205px; border-radius: 16px; display: grid; place-items: center;
            text-align: center; padding: 1.2rem; color: {MUTED};
            background: linear-gradient(135deg, #111A2A, #0A111D);
            border: 1px dashed rgba(140,155,179,.25);
        }}
        .clip-placeholder strong {{ display: block; color: {TEXT}; margin: .6rem 0 .2rem; }}
        .play-mark {{
            width: 48px; height: 48px; border-radius: 50%; display: grid; place-items: center;
            color: {BACKGROUND}; background: linear-gradient(135deg, {CYAN}, #72AFFF);
            margin: 0 auto; font-size: 1rem; box-shadow: 0 8px 30px rgba(81,229,212,.2);
        }}
        .neighbor-title {{ font-weight: 750; color: {TEXT}; margin-bottom: .15rem; }}
        .neighbor-meta {{ color: {MUTED}; font-size: .78rem; margin-bottom: .55rem; }}
        .callout {{
            padding: .9rem 1rem; border-left: 3px solid {PURPLE}; border-radius: 0 12px 12px 0;
            background: rgba(169,139,255,.07); color: #C9D2E1; font-size: .86rem; line-height: 1.5;
        }}
        hr {{ border-color: rgba(255,255,255,.07) !important; }}
        @media (max-width: 720px) {{
            .block-container {{ padding: 1.2rem 1rem 3rem; }}
            .hero {{ padding: 1.5rem; }} .live-pill {{ display: none; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, kicker: str) -> None:
    st.markdown(
        f'<div class="section-heading"><h2>{title}</h2><span>{kicker}</span></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-foot">{foot or '&nbsp;'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=28, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="Inter, ui-sans-serif, system-ui"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(bgcolor=PANEL_LIGHT, font_color=TEXT, bordercolor="#29364B"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def interpolate_curve(curve: list[dict[str, float]], hours: float, key: str) -> float:
    if not curve:
        return 0.0
    xs = [point["retained_hours"] for point in curve]
    index = bisect_left(xs, hours)
    if index <= 0:
        return curve[0][key]
    if index >= len(curve):
        return curve[-1][key]
    left, right = curve[index - 1], curve[index]
    span = right["retained_hours"] - left["retained_hours"]
    if span <= 0:
        return right[key]
    weight = (hours - left["retained_hours"]) / span
    return left[key] + (right[key] - left[key]) * weight


def display_clip(segment: dict[str, Any], key: str) -> None:
    clip_path = str(segment.get("clip_path") or "")
    is_remote = clip_path.startswith(("https://", "http://"))
    local_path = Path(clip_path) if clip_path else Path()
    if clip_path and not local_path.is_absolute():
        local_path = APP_DIR / local_path

    if is_remote:
        st.video(clip_path)
    elif clip_path and local_path.is_file():
        st.video(str(local_path))
    else:
        label = Path(clip_path).name if clip_path else "No clip_path supplied"
        st.markdown(
            f"""
            <div class="clip-placeholder" id="{key}">
              <div><div class="play-mark">▶</div><strong>Clip ready for pipeline media</strong>
              <span>{label}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def similar_segments(
    selected: dict[str, Any], all_segments: list[dict[str, Any]], limit: int = 5
) -> list[tuple[dict[str, Any], float, str]]:
    by_id = {segment["id"]: segment for segment in all_segments}
    output: list[tuple[dict[str, Any], float, str]] = []
    seen: set[str] = set()

    for relation in selected.get("neighbors", []):
        neighbor = by_id.get(relation["segment_id"])
        if not neighbor or neighbor["video_id"] == selected["video_id"]:
            continue
        explanation = relation.get("explanation") or (
            "Same annotation, different execution style"
            if neighbor["verb"] == selected["verb"]
            else "Visually similar motion pattern"
        )
        output.append((neighbor, relation["similarity"], explanation))
        seen.add(neighbor["id"])

    # Missing neighbor arrays are expected during schema iteration. This
    # deterministic mock fallback keeps the explorer useful without another
    # integration point; explicit pipeline neighbors always take precedence.
    candidates = [
        segment
        for segment in all_segments
        if segment["id"] != selected["id"]
        and segment["video_id"] != selected["video_id"]
        and segment["id"] not in seen
    ]
    candidates.sort(
        key=lambda segment: (
            segment["verb"] != selected["verb"],
            segment["cluster"] == selected["cluster"],
            abs(
                segment["distinctiveness_percentile"]
                - selected["distinctiveness_percentile"]
            ),
        )
    )
    for candidate in candidates:
        if len(output) >= limit:
            break
        same_verb = candidate["verb"] == selected["verb"]
        score = 0.95 if same_verb else 0.79
        score -= min(
            0.14,
            abs(
                candidate["distinctiveness_percentile"]
                - selected["distinctiveness_percentile"]
            )
            * 0.002,
        )
        explanation = (
            "Same annotation, different execution style"
            if same_verb and candidate["cluster"] != selected["cluster"]
            else "Same action family with a related motion signature"
        )
        output.append((candidate, max(0.0, score), explanation))

    return sorted(output, key=lambda item: item[1], reverse=True)[:limit]


def render_overview(
    summary: dict[str, Any], budget_hours: float, budget_coverage: float
) -> None:
    section_heading("Overview", "CURATION SNAPSHOT")
    original_hours = summary["original_hours"]
    reduction = (
        (1 - budget_hours / original_hours) * 100 if original_hours > 0 else 0.0
    )
    first_row = st.columns(4)
    cards = [
        ("Total videos", f"{summary['total_videos']:,}", "source corpus"),
        ("Total segments", f"{summary['total_segments']:,}", "action boundaries"),
        ("Original hours", f"{original_hours:.1f}h", "before curation"),
        ("Selected hours", f"{budget_hours:.1f}h", "active budget"),
    ]
    for column, (label, value, foot) in zip(first_row, cards):
        with column:
            metric_card(label, value, foot)

    second_row = st.columns(4)
    cards = [
        ("Data reduction", f"{max(0, reduction):.0f}%", "less footage to train on"),
        ("Behavioral coverage", f"{budget_coverage:.0f}%", "retained at active budget"),
        (
            "Composition diversity",
            f"{summary['composition_diversity']:.0f}",
            "action balance index",
        ),
        (
            "Execution diversity",
            f"{summary['execution_diversity']:.0f}",
            "style coverage index",
        ),
    ]
    for column, (label, value, foot) in zip(second_row, cards):
        with column:
            metric_card(label, value, foot)


def render_comparison(
    curve: list[dict[str, float]], summary: dict[str, Any]
) -> float:
    section_heading("Curation comparison", "DIVERSITY BEATS VOLUME")
    with st.container(border=True):
        left, right = st.columns([1.0, 2.2], gap="large")
        original_hours = max(summary["original_hours"], 0.1)
        curve_max = max(
            [point["retained_hours"] for point in curve] + [original_hours]
        )
        with left:
            st.markdown("### Training budget")
            budget = st.slider(
                "Hours retained",
                min_value=0.0,
                max_value=float(round(curve_max, 1)),
                value=float(
                    min(
                        curve_max,
                        st.session_state.get("budget_hours", summary["selected_hours"]),
                    )
                ),
                step=0.1,
                key="budget_hours",
            )
            retained_percent = min(100.0, budget / original_hours * 100)
            diversity = interpolate_curve(curve, budget, "diversity_coverage")
            random = interpolate_curve(curve, budget, "random_coverage")
            metric_card("Behavioral coverage", f"{diversity:.1f}%", f"+{max(0, diversity-random):.1f} pts vs random")
            st.caption(f"{retained_percent:.1f}% of source data · {budget:.1f} of {original_hours:.1f} hours")

        with right:
            if not curve:
                st.info("Add coverage_curve points to the results artifact to render this chart.")
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=[point["retained_percent"] for point in curve],
                        y=[point["diversity_coverage"] for point in curve],
                        name="Diversity selection",
                        mode="lines+markers",
                        line=dict(color=CYAN, width=4, shape="spline"),
                        marker=dict(size=7, color=CYAN, line=dict(width=2, color=PANEL)),
                        fill="tozeroy",
                        fillcolor="rgba(81,229,212,.07)",
                        hovertemplate="%{x:.0f}% retained<br>%{y:.1f}% coverage<extra>Diversity</extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[point["retained_percent"] for point in curve],
                        y=[point["random_coverage"] for point in curve],
                        name="Random selection",
                        mode="lines+markers",
                        line=dict(color=PURPLE, width=2.5, dash="dot", shape="spline"),
                        marker=dict(size=6, color=PURPLE),
                        hovertemplate="%{x:.0f}% retained<br>%{y:.1f}% coverage<extra>Random</extra>",
                    )
                )
                fig.add_vline(
                    x=retained_percent,
                    line_width=1.5,
                    line_dash="dash",
                    line_color=GOLD,
                    annotation_text=f"{budget:.1f}h budget",
                    annotation_font_color=GOLD,
                    annotation_position="top right",
                )
                fig.update_xaxes(title="Data retained", ticksuffix="%", range=[0, 102])
                fig.update_yaxes(title="Behavioral coverage", ticksuffix="%", range=[0, 105])
                st.plotly_chart(
                    chart_layout(fig, 410), width="stretch",
                    config={"displayModeBar": False},
                )
    return budget


def render_actions(actions: list[dict[str, Any]]) -> None:
    section_heading("Action composition", "BEFORE VS AFTER")
    if not actions:
        st.info("No action-composition data is available yet.")
        return
    preferred_order = [
        "fold", "pick", "smooth", "adjust", "straighten", "flip", "spread", "unfold"
    ]
    actions = sorted(
        actions,
        key=lambda item: (
            preferred_order.index(item["verb"])
            if item["verb"] in preferred_order
            else len(preferred_order),
            item["verb"],
        ),
    )
    before_total = sum(item["before"] for item in actions) or 1
    after_total = sum(item["after"] for item in actions) or 1
    labels = [item["verb"].title() for item in actions]
    before = [item["before"] / before_total * 100 for item in actions]
    after = [item["after"] / after_total * 100 for item in actions]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels, x=before, name="Original corpus", orientation="h",
            marker=dict(color="rgba(140,155,179,.38)", line=dict(color="#748299", width=1)),
            hovertemplate="%{y}<br>%{x:.1f}% of corpus<extra>Original</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels, x=after, name="Curated set", orientation="h",
            marker=dict(
                color=[VERB_COLORS.get(item["verb"], CYAN) for item in actions],
                line=dict(color="rgba(255,255,255,.4)", width=.5),
            ),
            hovertemplate="%{y}<br>%{x:.1f}% of curated set<extra>Curated</extra>",
        )
    )
    fig.update_layout(barmode="group", bargap=.30, bargroupgap=.08)
    fig.update_xaxes(title="Share of action segments", ticksuffix="%")
    fig.update_yaxes(autorange="reversed", title="")
    with st.container(border=True):
        st.plotly_chart(
            chart_layout(fig, 450), width="stretch",
            config={"displayModeBar": False},
        )


def render_explorer(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    section_heading("Segment explorer", "SEARCH · FILTER · SELECT")
    if not segments:
        st.info("No segments are present in the loaded results artifact.")
        return None

    verbs = sorted({segment["verb"] for segment in segments})
    episodes = sorted({segment["episode_id"] for segment in segments})
    filter_row = st.columns([1.5, 1, 1, 1.1, .9])
    with filter_row[0]:
        query = st.text_input(
            "Search", placeholder="ID, episode, verb or style…", key="segment_search"
        ).strip().lower()
    with filter_row[1]:
        selected_verbs = st.multiselect("Verb", verbs, placeholder="All verbs")
    with filter_row[2]:
        selected_episodes = st.multiselect("Episode", episodes, placeholder="All episodes")
    with filter_row[3]:
        score_range = st.slider("Score percentile", 0, 100, (0, 100))
    with filter_row[4]:
        status = st.selectbox("Decision", ["All", "Keep", "Drop"])

    filtered = []
    for segment in segments:
        searchable = " ".join(
            [segment["id"], segment["episode_id"], segment["verb"], segment["cluster"]]
        ).lower()
        if query and query not in searchable:
            continue
        if selected_verbs and segment["verb"] not in selected_verbs:
            continue
        if selected_episodes and segment["episode_id"] not in selected_episodes:
            continue
        if not score_range[0] <= segment["distinctiveness_percentile"] <= score_range[1]:
            continue
        if status == "Keep" and not segment["keep"]:
            continue
        if status == "Drop" and segment["keep"]:
            continue
        filtered.append(segment)

    table_rows = [
        {
            "Segment": segment["id"],
            "Episode": segment["episode_id"],
            "Verb": segment["verb"].title(),
            "Start": segment["start"],
            "End": segment["end"],
            "Cluster / style": segment["cluster"],
            "Distinctiveness": segment["distinctiveness_percentile"],
            "Decision": "KEEP" if segment["keep"] else "DROP",
        }
        for segment in filtered
    ]
    frame = pd.DataFrame(table_rows)
    st.caption(f"{len(filtered)} of {len(segments)} loaded segments · click a row to inspect it")
    if frame.empty:
        st.warning("No segments match the current filters.")
    else:
        event = st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            height=410,
            on_select="rerun",
            selection_mode="single-row",
            key="segment_table",
            column_config={
                "Start": st.column_config.NumberColumn(format="%.1f s"),
                "End": st.column_config.NumberColumn(format="%.1f s"),
                "Distinctiveness": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%"
                ),
                "Decision": st.column_config.TextColumn(width="small"),
            },
        )
        selection = getattr(event, "selection", None)
        rows = getattr(selection, "rows", []) if selection is not None else []
        if rows:
            row_index = rows[0]
            if 0 <= row_index < len(filtered):
                st.session_state.selected_segment_id = filtered[row_index]["id"]

    selected_id = st.session_state.get(
        "selected_segment_id", filtered[0]["id"] if filtered else segments[0]["id"]
    )
    return next(
        (segment for segment in segments if segment["id"] == selected_id),
        filtered[0] if filtered else segments[0],
    )


def render_segment_detail(
    selected: dict[str, Any] | None, segments: list[dict[str, Any]]
) -> None:
    section_heading("Segment detail", "NEAREST BEHAVIORAL NEIGHBORS")
    if selected is None:
        st.info("Select a segment in the explorer to inspect it.")
        return

    with st.container(border=True):
        main, metadata = st.columns([1.55, 1], gap="large")
        with main:
            display_clip(selected, f"primary-{selected['id']}")
        with metadata:
            st.markdown(f"### {selected['id']}")
            st.markdown(
                f"""
                <span class="badge keep">{'KEPT' if selected['keep'] else 'DROPPED'}</span>
                <span class="badge">{selected['verb'].upper()}</span>
                <span class="badge">{selected['cluster']}</span>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Episode**  ·  {selected['episode_id']}")
            st.markdown(f"**Window**  ·  {selected['start']:.1f}s — {selected['end']:.1f}s")
            st.markdown(
                f"**Distinctiveness**  ·  {selected['distinctiveness_percentile']:.0f}th percentile"
            )
            st.markdown(
                '<div class="callout">High-scoring segments preserve rare execution modes while avoiding redundant repetitions.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Five most similar segments")
    st.caption("Nearest matches are restricted to other videos.")
    neighbors = similar_segments(selected, segments)
    if not neighbors:
        st.info("No cross-video neighbors are available for this segment.")
        return
    for row_start in range(0, len(neighbors), 2):
        columns = st.columns(2, gap="large")
        for column, (neighbor, similarity, explanation) in zip(
            columns, neighbors[row_start : row_start + 2]
        ):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="neighbor-title">{neighbor["id"]} &nbsp;·&nbsp; {similarity:.0%} similar</div>'
                        f'<div class="neighbor-meta">{neighbor["episode_id"]} · {neighbor["cluster"]} · {neighbor["start"]:.1f}–{neighbor["end"]:.1f}s</div>',
                        unsafe_allow_html=True,
                    )
                    display_clip(neighbor, f"neighbor-{neighbor['id']}")
                    st.markdown(
                        f'<div class="callout">{explanation}</div>',
                        unsafe_allow_html=True,
                    )


def render_episode_view(
    segments: list[dict[str, Any]], selected: dict[str, Any] | None
) -> None:
    section_heading("Episode view", "FULL ACTION TIMELINE")
    if not segments:
        st.info("No episode data is available.")
        return
    episodes = sorted({segment["episode_id"] for segment in segments})
    default_episode = selected["episode_id"] if selected else episodes[0]
    default_index = episodes.index(default_episode) if default_episode in episodes else 0
    episode = st.selectbox("Episode", episodes, index=default_index, key="episode_view")
    episode_segments = sorted(
        [segment for segment in segments if segment["episode_id"] == episode],
        key=lambda segment: segment["start"],
    )

    fig = go.Figure()
    shown_verbs: set[str] = set()
    for segment in episode_segments:
        duration = max(0.01, segment["end"] - segment["start"])
        keep = segment["keep"]
        verb = segment["verb"]
        fig.add_trace(
            go.Bar(
                x=[duration],
                base=[segment["start"]],
                y=["Actions"],
                orientation="h",
                name=verb.title(),
                legendgroup=verb,
                showlegend=verb not in shown_verbs,
                width=.48 if keep else .34,
                opacity=1.0 if keep else .38,
                marker=dict(
                    color=VERB_COLORS.get(verb, MUTED),
                    line=dict(color=GOLD if keep else "rgba(255,255,255,.25)", width=3 if keep else 1),
                ),
                text=[verb.title() if duration >= 2.4 else ""],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=BACKGROUND if keep else TEXT, size=11),
                customdata=[
                    [segment["id"], segment["start"], segment["end"], "KEEP" if keep else "DROP", segment["cluster"]]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]:.1f}s–%{customdata[2]:.1f}s"
                    "<br>%{customdata[4]}<br><b>%{customdata[3]}</b><extra></extra>"
                ),
            )
        )
        shown_verbs.add(verb)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Episode time", ticksuffix="s", rangemode="tozero")
    fig.update_yaxes(title="", showgrid=False, fixedrange=True)
    with st.container(border=True):
        st.plotly_chart(
            chart_layout(fig, 290), width="stretch",
            config={"displayModeBar": False},
        )
        kept = sum(segment["keep"] for segment in episode_segments)
        st.caption(
            f"Gold outline = selected for retention · {kept} of {len(episode_segments)} segments kept"
        )


def main() -> None:
    st.set_page_config(
        page_title="EgoTrim · Behavioral Diversity",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme()
    results = load_results()
    summary = results["summary"]

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">EgoVerse · Behavioral intelligence</div>
          <h1>EgoTrim</h1>
          <p>A diversity-first view of what the dataset knows — and what we can remove without losing the behaviors that matter.</p>
          <div class="live-pill">● MOCK PIPELINE OUTPUT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if results["source_error"]:
        st.warning(results["source_error"])

    initial_budget = min(
        max(summary["selected_hours"], 0.0), max(summary["original_hours"], 0.1)
    )
    budget = float(st.session_state.get("budget_hours", initial_budget))
    budget_coverage = (
        interpolate_curve(results["coverage_curve"], budget, "diversity_coverage")
        if results["coverage_curve"]
        else summary["coverage_retained"]
    )
    render_overview(summary, budget, budget_coverage)
    render_comparison(results["coverage_curve"], summary)
    render_actions(results["actions"])
    selected = render_explorer(results["segments"])
    render_segment_detail(selected, results["segments"])
    render_episode_view(results["segments"], selected)

    st.markdown("---")
    st.caption(
        f"EgoTrim demo · results source: {Path(results['source_path']).name} · "
        "set EGOTRIM_RESULTS_PATH to load another artifact"
    )


if __name__ == "__main__":
    main()
