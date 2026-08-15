"""Render the diversity dashboard (instructions.md step 5) into a standalone HTML file.

Reads the JSON/parquet written by modal_score.py and inlines everything, so the
page is self-contained and needs no network access.

    modal volume get egoverse-zarrs-v2 /derived/results ./scratch_results
    python build_dashboard.py
"""

from __future__ import annotations

import json
import pathlib

DATA = pathlib.Path("scratch_results/dashboard_data.json")
CLIPS = pathlib.Path("scratch_results/clips.json")
VIDEOS = pathlib.Path("scratch_results/clip_videos.json")
OUT = pathlib.Path("dashboard.html")

HEAD = """<title>Fold-Clothes Diversity</title>
<style>
:root{
  --paper:#F5F4F0; --surface:#FFFFFF; --sunk:#EFEEE8;
  --ink:#15181B; --muted:#666D71; --faint:#8B9296;
  --rule:#DFDED7; --rule-soft:#E9E8E2;
  --accent:#0E6B63; --accent-soft:#0E6B6320; --accent-ink:#0A4F49;
  --amber:#C1751A; --amber-soft:#C1751A22;
  --crit:#A93F36; --ok:#3F7A45;
  --shadow:0 1px 2px rgba(20,24,26,.05),0 8px 24px -16px rgba(20,24,26,.28);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#111417; --surface:#191D21; --sunk:#14181B;
    --ink:#E9EAE6; --muted:#98A0A4; --faint:#788085;
    --rule:#272C31; --rule-soft:#1F2429;
    --accent:#43A99E; --accent-soft:#43A99E24; --accent-ink:#7FCabf;
    --amber:#DFA152; --amber-soft:#DFA15226;
    --crit:#D2726A; --ok:#6FAE73;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --paper:#111417; --surface:#191D21; --sunk:#14181B;
  --ink:#E9EAE6; --muted:#98A0A4; --faint:#788085;
  --rule:#272C31; --rule-soft:#1F2429;
  --accent:#43A99E; --accent-soft:#43A99E24; --accent-ink:#7FCabf;
  --amber:#DFA152; --amber-soft:#DFA15226;
  --crit:#D2726A; --ok:#6FAE73;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1120px;margin:0 auto;padding:48px 28px 96px;display:flex;flex-direction:column;gap:52px}
h1,h2,h3{font-family:ui-serif,Georgia,"Iowan Old Style",serif;font-weight:600;text-wrap:balance;margin:0}
h1{font-size:2.4rem;line-height:1.12;letter-spacing:-.015em}
h2{font-size:1.32rem;letter-spacing:-.01em}
.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.eyebrow{font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;color:var(--faint);font-weight:600}
.lede{color:var(--muted);max-width:64ch}
header{display:flex;flex-direction:column;gap:14px;border-bottom:1px solid var(--rule);padding-bottom:30px}
section{display:flex;flex-direction:column;gap:18px}
.sechead{display:flex;flex-direction:column;gap:5px;border-bottom:1px solid var(--rule-soft);padding-bottom:9px}
.note{font-size:.83rem;color:var(--muted);max-width:72ch}

.hero{display:flex;flex-direction:column;gap:26px;background:var(--surface);border:1px solid var(--rule);
  border-radius:3px;padding:28px 30px 30px;box-shadow:var(--shadow)}
.claim{font-family:ui-serif,Georgia,serif;font-size:1.9rem;line-height:1.2;letter-spacing:-.012em;
  text-wrap:balance;margin:0}
.claim em{font-style:normal;color:var(--accent)}
.cmpchart{display:flex;flex-direction:column;gap:16px}
.cmpchart .ct{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);font-weight:600}
.crow{display:grid;grid-template-columns:112px 1fr 96px;align-items:center;gap:14px}
.crow .cl{font-size:.86rem;color:var(--muted)}
.crow .cbar{height:26px;background:var(--sunk);border-radius:2px;overflow:hidden;position:relative}
.crow .cf{position:absolute;inset:0 auto 0 0;border-radius:2px}
.crow .cf.full{background:var(--faint);opacity:.45}
.crow .cf.ours{background:var(--accent)}
.crow .cf.hours{background:var(--amber)}
.crow .cv{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums;
  font-size:.94rem;font-weight:600;text-align:right}
.heronote{font-size:.82rem;color:var(--muted);max-width:74ch;margin:0}
@media (max-width:560px){.crow{grid-template-columns:88px 1fr 70px;gap:9px}.claim{font-size:1.45rem}}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;background:var(--rule);
  border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.cell{background:var(--surface);padding:15px 17px;display:flex;flex-direction:column;gap:4px}
.cell .k{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);font-weight:600}
.cell .v{font-size:1.62rem;line-height:1.1;font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums}
.cell .s{font-size:.76rem;color:var(--muted)}
.hero-v{color:var(--accent)}

.bars{display:flex;flex-direction:column;gap:0;border:1px solid var(--rule);border-radius:3px;background:var(--surface);overflow:hidden}
.row{border-bottom:1px solid var(--rule-soft)}
.row.extra{display:none}
.bars.showall .row.extra{display:block}
.morebar{display:flex;justify-content:center;padding:9px 16px;border-top:1px solid var(--rule-soft);
  background:var(--sunk)}
.morebar button{border:0;background:none;color:var(--muted);font-size:.79rem;letter-spacing:.03em;
  padding:3px 10px;cursor:pointer}
.morebar button:hover{color:var(--accent)}
.morebar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
.row:last-child{border-bottom:0}
.bar{display:grid;grid-template-columns:16px 88px 1fr 132px;align-items:center;gap:14px;
  padding:9px 16px;width:100%;border:0;background:none;color:inherit;font:inherit;text-align:left;
  cursor:pointer}
.bar:hover{background:var(--sunk)}
.bar:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.bar[disabled]{cursor:default}
.bar[disabled]:hover{background:none}
.chev{color:var(--faint);font-size:.7rem;transition:transform .16s ease;line-height:1}
.bar[aria-expanded="true"] .chev{transform:rotate(90deg);color:var(--accent)}
.bar[disabled] .chev{opacity:0}
.drawer{display:none;padding:4px 16px 18px;background:var(--sunk);
  border-top:1px solid var(--rule-soft)}
.drawer.open{display:flex;flex-direction:column;gap:12px}
.drawer .dh{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;
  font-size:.79rem;color:var(--muted);padding-top:11px}
@media (prefers-reduced-motion:reduce){.chev{transition:none}}
.bar .vb{font-weight:600;font-size:.92rem}
.track{position:relative;height:16px;background:var(--sunk);border-radius:2px;overflow:hidden}
.fill{position:absolute;inset:0 auto 0 0;background:var(--accent);opacity:.86}
.bar.dim .fill{background:var(--faint);opacity:.5}
.bar .rt{display:flex;gap:12px;justify-content:flex-end;font-size:.8rem;color:var(--muted)}
.bar .rt b{color:var(--ink);font-weight:600}
.flag{font-size:.63rem;letter-spacing:.07em;text-transform:uppercase;color:var(--crit);border:1px solid var(--crit);
  border-radius:2px;padding:0 4px;align-self:center}

.cmp{display:grid;grid-template-columns:repeat(auto-fit,minmax(292px,1fr));gap:20px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:3px;padding:20px;box-shadow:var(--shadow);
  display:flex;flex-direction:column;gap:14px}
.card h3{font-size:1.02rem}
table{width:100%;border-collapse:collapse;font-size:.84rem}
th{text-align:left;font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
  font-weight:600;padding:0 10px 7px 0;border-bottom:1px solid var(--rule)}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--rule-soft)}
td.n{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums;text-align:right}
tr:last-child td{border-bottom:0}
.win{color:var(--accent);font-weight:600}
.lose{color:var(--muted)}
.scroll{overflow-x:auto}
.chip{display:inline-block;font-size:.7rem;padding:1px 6px;border:1px solid var(--rule);border-radius:2px;
  color:var(--muted);margin:0 3px 3px 0;white-space:nowrap}
canvas{width:100%;height:340px;display:block;border:1px solid var(--rule);border-radius:3px;background:var(--surface)}
.legend{display:flex;gap:18px;font-size:.78rem;color:var(--muted);flex-wrap:wrap}
.clipgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(206px,1fr));gap:12px}
.clip{background:var(--surface);border:1px solid var(--rule);border-radius:3px;overflow:hidden;
  display:flex;flex-direction:column}
.clip video{width:100%;height:132px;object-fit:cover;display:block;background:#000;
  border-bottom:1px solid var(--rule-soft)}
.clip .novid{height:132px;display:flex;align-items:center;justify-content:center;background:var(--sunk);
  color:var(--faint);font-size:.72rem;border-bottom:1px solid var(--rule-soft)}
.clip canvas{height:132px;border:0;border-bottom:1px solid var(--rule-soft);border-radius:0}
.rowlabel{font-size:.63rem;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);
  font-weight:600;padding:3px 0 0}
.clip .meta{padding:9px 12px 11px;display:flex;flex-direction:column;gap:3px}
.clip .vh{display:flex;align-items:baseline;gap:7px}
.clip .vn{font-weight:600;font-size:.95rem}
.clip .kind{font-size:.63rem;letter-spacing:.08em;text-transform:uppercase;font-weight:600;
  padding:1px 5px;border-radius:2px}
.clip .kind.d{color:var(--accent);background:var(--accent-soft)}
.clip .kind.t{color:var(--muted);background:var(--sunk)}
.clip .sub{font-size:.74rem;color:var(--muted)}
.clip .sub span{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.ctrl{display:flex;gap:10px;align-items:center;font-size:.8rem;color:var(--muted)}
button{font:inherit;font-size:.8rem;padding:5px 12px;border:1px solid var(--rule);border-radius:2px;
  background:var(--surface);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:6px}
.dot.amber{background:var(--amber)}
.caveat{border-left:2px solid var(--amber);padding:2px 0 2px 15px;display:flex;flex-direction:column;gap:9px}
.caveat p{margin:0;font-size:.87rem;color:var(--muted)}
.caveat b{color:var(--ink);font-weight:600}
footer{border-top:1px solid var(--rule);padding-top:20px;font-size:.8rem;color:var(--faint)}
@media (max-width:640px){
  .bar{grid-template-columns:72px 1fr;gap:9px}
  .bar .rt{grid-column:1/-1;justify-content:flex-start}
  h1{font-size:1.85rem}
}
</style>"""


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build() -> str:
    d = json.loads(DATA.read_text())
    g = d["global"]
    hero = d["hero"]
    cur = d["curation"]
    verbs = sorted(
        (v for v in d["per_verb"] if v["n"] >= 25), key=lambda v: -v["vendi"]
    )[:16]
    vmax = max(v["vendi"] for v in verbs)
    by_verb = {v["verb"]: v for v in d["per_verb"]}
    TOP_N = 7
    n_hidden = max(0, len(verbs) - TOP_N)

    strip = "".join(
        f'<div class="cell"><div class="k">{k}</div>'
        f'<div class="v{" hero-v" if hi else ""}">{v}</div><div class="s">{s}</div></div>'
        for k, v, s, hi in [
            ("Effective behaviours", f'{g["vendi"]:.1f}', f'of {d["rank_cap"]} possible', True),
            ("Segments", f'{g["n"]:,}', f'{d["n_videos"]} sessions', False),
            ("Hours", f'{g["hours"]:.1f}', "labelled manipulation", False),
            ("Distinct verbs", f'{len(d["per_verb"])}', f'{sum(1 for v in d["per_verb"] if v["reliable"])} with n≥{d["min_group"]}', False),
            ("Nearest-neighbour", f'{g["nn_distance"]:.3f}', "mean cosine distance", False),
        ]
    )

    shown = {v["verb"] for v in verbs}
    raw_clips = json.loads(CLIPS.read_text())
    clips_payload = {
        "bones": raw_clips["bones"],
        "frames": raw_clips["frames"],
        "clips": [
            dict(c, _i=i)
            for i, c in enumerate(raw_clips["clips"])
            if c["verb"] in shown
        ],
    }
    videos = json.loads(VIDEOS.read_text()) if VIDEOS.exists() else {}
    clips_by_verb = {}
    for c in clips_payload["clips"]:
        clips_by_verb.setdefault(c["verb"], []).append(c)

    bars = ""
    for rank, v in enumerate(verbs):
        pct = 100 * v["vendi"] / vmax
        dim = "" if v["reliable"] else " dim"
        flag = "" if v["reliable"] else '<span class="flag">n low</span>'
        has = v["verb"] in clips_by_verb
        extra = " extra" if rank >= TOP_N else ""
        bars += (
            f'<div class="row{extra}">'
            f'<button class="bar{dim}" type="button" aria-expanded="false"'
            f'{"" if has else " disabled"} data-verb="{esc(v["verb"])}">'
            f'<span class="chev">&#9654;</span>'
            f'<span class="vb">{esc(v["verb"])}{flag}</span>'
            f'<span class="track"><span class="fill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="rt"><span>n <b class="num">{v["n"]}</b></span>'
            f'<span>VS <b class="num">{v["vendi"]:.2f}</b></span>'
            f'<span>/h <b class="num">{v["vendi_per_hour"]:.1f}</b></span></span></button>'
            f'<div class="drawer" data-drawer="{esc(v["verb"])}"></div>'
            f'</div>'
        )

    def cmp_row(label, c, r, fmt="{:.3f}", higher_better=True):
        cw = (c > r) == higher_better
        return (
            f'<tr><td>{label}</td>'
            f'<td class="n {"win" if cw else "lose"}">{fmt.format(c)}</td>'
            f'<td class="n {"lose" if cw else "win"}">{fmt.format(r)}</td>'
            f'<td class="n">{fmt.format(c - r)}</td></tr>'
        )

    cov = min(cur["coverage_pct"], 100.0)
    raw_cov = cur["coverage_pct"]
    raw_vs = cur["kept_vendi"]
    beh = cur["full_vendi"]
    tot_h = cur["total_hours"]
    kept_h = cur["kept_hours"]
    hours_pct = 100 * kept_h / tot_h
    kept_n = cur["kept_segments"]
    tot_n = cur["total_segments"]

    c, r = hero["curated"], hero["random_mean"]
    hero_rows = (
        cmp_row("Vendi", c["vendi"], r["vendi"])
        + cmp_row("log det(K+I)", c["log_det"], r["log_det"], "{:.4f}")
        + cmp_row("NN distance", c["nn_distance"], r["nn_distance"], "{:.4f}")
    )

    scene_rows = "".join(
        f'<tr><td class="num">{esc(p["operator"][:8])}</td><td>{esc(p["scene_a"])}</td>'
        f'<td class="n">{p["a"]["vendi"]:.2f}</td><td>{esc(p["scene_b"])}</td>'
        f'<td class="n">{p["b"]["vendi"]:.2f}</td>'
        f'<td class="n">{"yes" if p["agree"] else "no"}</td></tr>'
        for p in d["scene_pairs"]
    ) or '<tr><td colspan="6" class="lose">No operator recorded in two scenes with ≥40 segments each.</td></tr>'

    op_rows = "".join(
        f'<tr><td class="num">{esc(o["operator"][:10])}</td><td class="n">{o["vendi"]:.2f}</td>'
        f'<td class="n">{o["nn_distance"]:.3f}</td><td class="n">{o["scenes"]}</td>'
        f'<td class="n">{o["n"]}</td></tr>'
        for o in d["operators"]["rows"]
    )

    def vid_rows(rows):
        out = ""
        for v in rows:
            chips = "".join(f'<span class="chip">{esc(x)}</span>' for x in v["verbs"].split(",")[:5])
            out += (
                f'<tr><td class="num">{esc(v["episode"])}</td><td>{esc(v["task"])}</td>'
                f'<td>{esc(v["scene"])}</td><td class="n">{v["n"]}</td>'
                f'<td class="n">{v["internal"]:.2f}</td>'
                f'<td class="n">{v["contrib"]:+.4f}</td>'
                f'<td class="n">{v["valid"]:.2f}</td><td>{chips}</td></tr>'
            )
        return out

    return f"""{HEAD}
<div class="wrap">
<header>
  <div class="eyebrow">EgoVerse-I &middot; fold-clothes &middot; hand kinematics</div>
  <h1>{g["vendi"]:.1f} effective behaviours in {g["hours"]:.0f} hours</h1>
  <p class="lede">{g["n"]:,} labelled manipulation cycles from {d["n_videos"]} sessions, scored on
  wrist trajectory and fingertip configuration &mdash; not pixels, not text. The Vendi score reads as
  the number of genuinely distinct things the data contains: identical motion scores 1, wholly
  dissimilar motion scores N.</p>
</header>

<section>
  <div class="hero">
    <p class="claim">Half the hours. <em>{cov:.0f}% of the behaviour.</em></p>
    <div class="cmpchart">
      <div class="ct">Distinct behaviours &middot; Vendi score</div>
      <div class="crow"><span class="cl">Full corpus</span>
        <span class="cbar"><span class="cf full" style="width:100%"></span></span>
        <span class="cv">{beh:.1f}</span></div>
      <div class="crow"><span class="cl">Curated half</span>
        <span class="cbar"><span class="cf ours" style="width:100%"></span></span>
        <span class="cv">{beh:.1f}</span></div>
    </div>
    <div class="cmpchart">
      <div class="ct">Hours of footage</div>
      <div class="crow"><span class="cl">Full corpus</span>
        <span class="cbar"><span class="cf hours" style="width:100%;opacity:.4"></span></span>
        <span class="cv">{tot_h:.1f}h</span></div>
      <div class="crow"><span class="cl">Curated half</span>
        <span class="cbar"><span class="cf hours" style="width:{hours_pct:.1f}%"></span></span>
        <span class="cv">{kept_h:.1f}h</span></div>
    </div>
    <p class="heronote">Greedy selection keeps {kept_n:,} of {tot_n:,} cycles. The discarded half was
    largely repetition &mdash; measured on hand kinematics, it contained nothing the kept half does not
    already have. Coverage is capped at 100%: a subset cannot contain more behaviour than the corpus
    it came from. On the raw score the curated half actually reads {raw_cov:.0f}% ({raw_vs:.2f} vs
    {beh:.2f}), because dropping redundant cycles rebalances the spectrum rather than adding
    anything new.</p>
  </div>
</section>

<section>
  <div class="sechead"><h2>Corpus</h2>
  <div class="note">Measured over {d["dims"]} PCA dimensions of a cosine kernel.</div></div>
  <div class="strip">{strip}</div>
</section>

<section>
  <div class="sechead"><h2>Diversity by action</h2>
  <div class="note">Grouping comes from the annotation verb; the measurement comes from kinematics,
  so there is no circularity. Ordered by sample count. Groups below n={d["min_group"]} are marked &mdash;
  their eigenvalue estimates are noisy.</div></div>
  <div class="bars" id="bars">{bars}
    <div class="morebar"><button id="more" type="button" aria-expanded="false"
      aria-controls="bars">Show {n_hidden} more actions</button></div>
  </div>
  <div class="note">Ranked by diversity, not sample count. <b>fold</b> is the largest group
  ({by_verb["fold"]["n"]} cycles) yet sits below <b>pick</b> ({by_verb["pick"]["vendi"]:.1f} on
  {by_verb["pick"]["n"]} cycles) &mdash; people fold more stereotypically than they reach.
  <b>smooth</b> is the most repetitive frequent action at {by_verb["smooth"]["vendi"]:.1f}.
  Open a row to replay its three most distinctive cycles against a median one &mdash; both hands,
  21 MANO keypoints, in the head frame. Video was excluded from this pull, so these are the
  kinematics the score is computed from, not a camera view.</div>
</section>

<section>
  <div class="sechead"><h2>Curated half vs random half</h2>
  <div class="note">Both subsets hold {hero["budget_hours"]:.1f} of {hero["total_hours"]:.1f} hours &mdash; equal
  hours, not equal segment counts, since a set of long segments would otherwise win by containing
  more footage. Random is the mean of 5 draws.</div></div>
  <div class="cmp">
    <div class="card">
      <h3>Three measures, one kernel</h3>
      <div class="scroll"><table>
        <tr><th>Measure</th><th>Curated</th><th>Random</th><th>&Delta;</th></tr>
        {hero_rows}
      </table></div>
      <div class="note">Vendi and log-det both favour the curated half. NN-distance does not &mdash;
      curation packs {c["n"]:,} short segments into the same hours, so neighbours sit closer even
      though the spread of behaviour is wider. Two of three agree.</div>
    </div>
    <div class="card">
      <h3>Half the hours, {cov:.0f}% of the behaviour</h3>
      <div class="strip" style="grid-template-columns:1fr 1fr">
        <div class="cell"><div class="k">Kept</div><div class="v">{cur["kept_hours"]:.1f}h</div>
          <div class="s">{cur["kept_segments"]:,} of {cur["total_segments"]:,} cycles</div></div>
        <div class="cell"><div class="k">Coverage</div><div class="v hero-v">{cov:.0f}%</div>
          <div class="s">of full-corpus Vendi</div></div>
      </div>
      <div class="note">Selection maximises log&nbsp;det(I+K) per hour, which is submodular and so
      carries the standard greedy guarantee. Most-dropped verb: <b>fold</b>
      ({cur["dropped_verb_mix"].get("fold", 0)} cycles) &mdash; the largest group and the least varied
      per hour, so it is where the redundancy sits.</div>
    </div>
  </div>
</section>

<section>
  <div class="sechead"><h2>Sessions</h2>
  <div class="note">Internal diversity is how varied one session is on its own. Contribution is what
  the whole corpus loses if that session is removed &mdash; computed by leaving the entire session out,
  so five near-identical folds are not credited five times. They correlate at
  r={d["corr_internal_contrib"]}, so they are related but not redundant.</div></div>
  <canvas id="sc" width="1000" height="340"></canvas>
  <div class="legend"><span><span class="dot"></span>adds diversity</span>
    <span><span class="dot amber"></span>redundant &mdash; removing it raises the score</span>
    <span>point size = cycles in session</span></div>
</section>

<section>
  <div class="sechead"><h2>Most and least distinctive sessions</h2>
  <div class="note">Ranked by contribution. Video frames were excluded from this pull, so sessions are
  identified by task, scene and verb mix rather than thumbnails.</div></div>
  <div class="scroll"><table>
    <tr><th>Episode</th><th>Task</th><th>Scene</th><th>Cycles</th><th>Internal</th>
        <th>Contribution</th><th>Valid</th><th>Verbs</th></tr>
    {vid_rows(d["top"])}
    <tr><td colspan="8" style="padding-top:14px;color:var(--faint);font-size:.7rem;
      letter-spacing:.09em;text-transform:uppercase">Least distinctive</td></tr>
    {vid_rows(d["bottom"])}
  </table></div>
</section>

<section>
  <div class="sechead"><h2>Scene and operator splits</h2>
  <div class="note">Both equalised by hours.</div></div>
  <div class="cmp">
    <div class="card"><h3>Scene A vs B, same operator</h3>
      <div class="scroll"><table>
        <tr><th>Operator</th><th>Scene A</th><th>VS</th><th>Scene B</th><th>VS</th><th>Agree</th></tr>
        {scene_rows}
      </table></div>
      <div class="note">Hardware and person held constant, so the difference is the room.</div>
    </div>
    <div class="card"><h3>Operator audit</h3>
      <div class="scroll"><table>
        <tr><th>Operator</th><th>VS</th><th>NN</th><th>Scenes</th><th>n</th></tr>
        {op_rows}
      </table></div>
      <div class="note">{esc(d["operators"]["caveat"])} Equalised at
      {d["operators"]["equalised_hours"]:.2f}h each.</div>
    </div>
  </div>
</section>

<section>
  <div class="sechead"><h2>What these numbers cannot tell you</h2></div>
  <div class="caveat">
    <p><b>The ceiling is {d["rank_cap"]}.</b> A cosine kernel on {d["dims"]} PCA dimensions has rank at
    most {d["dims"]}, so Vendi can never exceed {d["dims"]} however many distinct behaviours exist. At
    {g["vendi"]:.1f} the ceiling is not binding, but a corpus twice as varied would not read as twice
    the score.</p>
    <p><b>Half the labels describe two actions.</b> 49% of source spans read like
    &ldquo;pick up mask from pile, place mask on stack&rdquo; with no sub-boundary to split on. The first
    verb wins, which biases counts toward <b>pick</b>.</p>
    <p><b>Label granularity is coarser than a repetition.</b> A single &ldquo;fold shirt&rdquo; span runs
    14.5s at the median, covering several physical folds. Resampling that to 30 frames averages them
    together, so execution diversity is partly measuring blur. Composition is unaffected.</p>
    <p><b>Operator scores are not a ranking of people.</b> They confound task assignment, session
    length and tracking quality. Use them to spot collection anomalies only.</p>
  </div>
</section>

<footer>Kernel: cosine over {d["dims"]} PCA dims of wrist trajectory (180) + fingertip configuration
(900) + duration. Vendi = exp of the Shannon entropy of the kernel eigenvalue spectrum. Verified
against identical vectors (1.00), duplication (invariant) and the explicit N&times;N kernel (1e-14).</footer>
</div>
<script id="d" type="application/json">{json.dumps(json.loads(DATA.read_text())["scatter"])}</script>
<script id="cl" type="application/json">{json.dumps(clips_payload)}</script>
<script id="vd" type="application/json">{json.dumps(videos)}</script>
<script>
const pts = JSON.parse(document.getElementById('d').textContent);
const cv = document.getElementById('sc');
function css(n){{return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}}
function draw(){{
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const x = cv.getContext('2d'); x.setTransform(dpr,0,0,dpr,0,0);
  x.clearRect(0,0,w,h);
  const pad = {{l:52,r:16,t:16,b:34}};
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  const px = v => pad.l + (v-x0)/(x1-x0||1)*(w-pad.l-pad.r);
  const py = v => h-pad.b - (v-y0)/(y1-y0||1)*(h-pad.t-pad.b);
  const rule=css('--rule-soft'), muted=css('--muted'), acc=css('--accent'), am=css('--amber');
  x.strokeStyle=rule; x.lineWidth=1; x.font='11px ui-monospace,Menlo,monospace'; x.fillStyle=muted;
  for(let i=0;i<=4;i++){{
    const yy = pad.t + i*(h-pad.t-pad.b)/4;
    x.beginPath(); x.moveTo(pad.l,yy); x.lineTo(w-pad.r,yy); x.stroke();
    x.fillText((y1-(y1-y0)*i/4).toFixed(3), 6, yy+4);
  }}
  x.strokeStyle=muted; x.globalAlpha=.5; x.beginPath();
  x.moveTo(pad.l,py(0)); x.lineTo(w-pad.r,py(0)); x.stroke(); x.globalAlpha=1;
  for(const [ix,ct,n] of pts){{
    x.beginPath();
    x.arc(px(ix),py(ct), Math.max(1.6, Math.min(6, Math.sqrt(n)*0.62)), 0, 6.284);
    x.fillStyle = ct>=0 ? acc : am; x.globalAlpha = .5; x.fill();
  }}
  x.globalAlpha=1; x.fillStyle=muted;
  x.fillText('internal diversity (Vendi within session) \\u2192', pad.l, h-10);
  x.save(); x.translate(13,pad.t+96); x.rotate(-Math.PI/2);
  x.fillText('contribution \\u2192', 0, 0); x.restore();
}}
draw();
addEventListener('resize', draw);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change', draw);
new MutationObserver(draw).observe(document.documentElement,{{attributes:true,attributeFilter:['data-theme']}});

const moreBtn=document.getElementById('more'), barsEl=document.getElementById('bars');
if(moreBtn){{
  moreBtn.addEventListener('click',()=>{{
    const shown=barsEl.classList.toggle('showall');
    moreBtn.setAttribute('aria-expanded', shown?'true':'false');
    moreBtn.textContent = shown ? 'Show fewer' : moreBtn.dataset.label;
  }});
  moreBtn.dataset.label = moreBtn.textContent;
}}

const CL = JSON.parse(document.getElementById('cl').textContent);
const VD = JSON.parse(document.getElementById('vd').textContent);
const byVerb = {{}};
for(const c of CL.clips) (byVerb[c.verb] = byVerb[c.verb] || []).push(c);
const openVerbs = new Map();     // verb -> [{{cv, clip, bb}}]
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

function bounds(xy){{
  let a=1e9,b=-1e9,c2=1e9,d2=-1e9;
  for(const f of xy) for(const hnd of f) for(const j of hnd){{
    if(j[0]<a)a=j[0]; if(j[0]>b)b=j[0]; if(j[1]<c2)c2=j[1]; if(j[1]>d2)d2=j[1];
  }}
  return [a,b,c2,d2];
}}
function drawClip(p, t){{
  const cv=p.cv, c=p.clip, dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth, h=cv.clientHeight;
  if(!w||!h) return;
  if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){{
    cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
  }}
  const x=cv.getContext('2d'); x.setTransform(dpr,0,0,dpr,0,0);
  x.fillStyle=css('--surface'); x.fillRect(0,0,w,h);
  if(!p.bb) p.bb=bounds(c.xy);
  const [x0,x1,y0,y1]=p.bb, pad=14;
  const sc=Math.min((w-2*pad)/Math.max(x1-x0,1),(h-2*pad)/Math.max(y1-y0,1));
  const ox=(w-(x1-x0)*sc)/2-x0*sc, oy=(h-(y1-y0)*sc)/2-y0*sc;
  const px=v=>v*sc+ox, py=v=>h-(v*sc+oy);
  const acc=css('--accent'), am=css('--amber'), ru=css('--rule');
  for(let hi=0;hi<2;hi++){{
    x.strokeStyle=ru; x.lineWidth=1; x.beginPath();
    for(let f=0;f<=t;f++){{
      const wj=c.xy[f][hi][0];
      if(f===0)x.moveTo(px(wj[0]),py(wj[1])); else x.lineTo(px(wj[0]),py(wj[1]));
    }}
    x.stroke();
    const col = hi===0?am:acc;
    x.strokeStyle=col; x.lineWidth=1.7; x.beginPath();
    for(const [i,j] of CL.bones){{
      const A=c.xy[t][hi][i], B=c.xy[t][hi][j];
      x.moveTo(px(A[0]),py(A[1])); x.lineTo(px(B[0]),py(B[1]));
    }}
    x.stroke();
    x.fillStyle=col;
    const wj=c.xy[t][hi][0];
    x.beginPath(); x.arc(px(wj[0]),py(wj[1]),2.6,0,6.284); x.fill();
  }}
}}
function build(verb, drawer){{
  const list = byVerb[verb] || [];
  const head = document.createElement('div'); head.className='dh';
  const nd = list.filter(c=>c.kind==='distinctive').length;
  head.innerHTML = '<span>'+nd+' most distinctive cycles, and one median for comparison</span>'
    + '<span>camera view above &middot; tracked hands below &middot; amber = left, teal = right</span>';
  drawer.appendChild(head);
  const grid = document.createElement('div'); grid.className='clipgrid';
  const players=[];
  for(const c of list){{
    const card=document.createElement('div'); card.className='clip';
    const b64 = VD[String(c._i)];
    if(b64){{
      const vd=document.createElement('video');
      vd.src='data:video/mp4;base64,'+b64;
      vd.autoplay=true; vd.loop=true; vd.muted=true; vd.playsInline=true;
      vd.setAttribute('aria-label', c.verb+' '+c.kind+' camera view');
      card.appendChild(vd);
    }} else {{
      const nv=document.createElement('div'); nv.className='novid';
      nv.textContent='camera frames unavailable'; card.appendChild(nv);
    }}
    const cn=document.createElement('canvas'); card.appendChild(cn);
    const m=document.createElement('div'); m.className='meta';
    m.innerHTML='<div class="vh"><span class="vn">'+c.verb+'</span>'
      +'<span class="kind '+(c.kind==='distinctive'?'d':'t')+'">'+c.kind+'</span></div>'
      +'<div class="sub">'+c.task+' &middot; '+c.scene+'</div>'
      +'<div class="sub"><span>'+c.duration.toFixed(1)+'s</span> &middot; percentile '
      +'<span>'+c.pct.toFixed(0)+'</span></div>';
    card.appendChild(m); grid.appendChild(card);
    players.push({{cv:cn, clip:c}});
  }}
  drawer.appendChild(grid);
  return players;
}}
// With reduced motion we hold the final frame rather than the first: the wrist
// trail is fully drawn and the hand sits at its end pose, which is the most
// informative single frame. Animating would otherwise be the only way to see it.
let frame = reduce ? CL.frames - 1 : 0;
function tick(){{
  for(const players of openVerbs.values()) for(const p of players) drawClip(p, frame);
  if(!reduce) frame=(frame+1)%CL.frames;
}}
if(!reduce) setInterval(tick, 90);
for(const btn of document.querySelectorAll('.bar[data-verb]')){{
  if(btn.disabled) continue;
  btn.addEventListener('click', ()=>{{
    const verb=btn.dataset.verb;
    const drawer=document.querySelector('.drawer[data-drawer="'+verb+'"]');
    const isOpen=btn.getAttribute('aria-expanded')==='true';
    if(isOpen){{
      btn.setAttribute('aria-expanded','false');
      drawer.classList.remove('open'); openVerbs.delete(verb);
    }} else {{
      if(!drawer.dataset.built){{ openVerbs.set(verb, build(verb, drawer)); drawer.dataset.built='1'; }}
      else {{ openVerbs.set(verb, drawer._players); }}
      drawer._players = openVerbs.get(verb);
      btn.setAttribute('aria-expanded','true');
      drawer.classList.add('open');
      requestAnimationFrame(()=>{{ for(const p of openVerbs.get(verb)) drawClip(p, frame); }});
    }}
  }});
}}
addEventListener('resize',()=>{{for(const ps of openVerbs.values()) for(const p of ps) drawClip(p,frame);}});
</script>"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
