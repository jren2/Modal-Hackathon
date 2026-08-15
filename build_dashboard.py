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

.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;background:var(--rule);
  border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.cell{background:var(--surface);padding:15px 17px;display:flex;flex-direction:column;gap:4px}
.cell .k{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);font-weight:600}
.cell .v{font-size:1.62rem;line-height:1.1;font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums}
.cell .s{font-size:.76rem;color:var(--muted)}
.hero-v{color:var(--accent)}

.bars{display:flex;flex-direction:column;gap:0;border:1px solid var(--rule);border-radius:3px;background:var(--surface);overflow:hidden}
.bar{display:grid;grid-template-columns:88px 1fr 132px;align-items:center;gap:14px;padding:9px 16px;border-bottom:1px solid var(--rule-soft)}
.bar:last-child{border-bottom:0}
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
    verbs = [v for v in d["per_verb"] if v["n"] >= 25][:16]
    vmax = max(v["vendi"] for v in verbs)

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

    bars = ""
    for v in verbs:
        pct = 100 * v["vendi"] / vmax
        dim = "" if v["reliable"] else " dim"
        flag = "" if v["reliable"] else '<span class="flag">n low</span>'
        bars += (
            f'<div class="bar{dim}"><div class="vb">{esc(v["verb"])}{flag}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="rt"><span>n <b class="num">{v["n"]}</b></span>'
            f'<span>VS <b class="num">{v["vendi"]:.2f}</b></span>'
            f'<span>/h <b class="num">{v["vendi_per_hour"]:.0f}</b></span></div></div>'
        )

    def cmp_row(label, c, r, fmt="{:.3f}", higher_better=True):
        cw = (c > r) == higher_better
        return (
            f'<tr><td>{label}</td>'
            f'<td class="n {"win" if cw else "lose"}">{fmt.format(c)}</td>'
            f'<td class="n {"lose" if cw else "win"}">{fmt.format(r)}</td>'
            f'<td class="n">{fmt.format(c - r)}</td></tr>'
        )

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
  <div class="sechead"><h2>Corpus</h2>
  <div class="note">Measured over {d["dims"]} PCA dimensions of a cosine kernel.</div></div>
  <div class="strip">{strip}</div>
</section>

<section>
  <div class="sechead"><h2>Diversity by action</h2>
  <div class="note">Grouping comes from the annotation verb; the measurement comes from kinematics,
  so there is no circularity. Ordered by sample count. Groups below n={d["min_group"]} are marked &mdash;
  their eigenvalue estimates are noisy.</div></div>
  <div class="bars">{bars}</div>
  <div class="note"><b>fold</b> is the largest group ({verbs[0]["n"]} cycles) yet scores
  {verbs[0]["vendi"]:.1f}, below <b>pick</b> at {verbs[1]["vendi"]:.1f} on fewer samples &mdash; people fold more
  stereotypically than they reach. <b>smooth</b> is the most repetitive frequent action at
  {[v for v in verbs if v["verb"]=="smooth"][0]["vendi"]:.1f}.</div>
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
      <h3>{cur["headline"]}</h3>
      <div class="strip" style="grid-template-columns:1fr 1fr">
        <div class="cell"><div class="k">Kept</div><div class="v">{cur["kept_hours"]:.1f}h</div>
          <div class="s">{cur["kept_segments"]:,} of {cur["total_segments"]:,} cycles</div></div>
        <div class="cell"><div class="k">Coverage</div><div class="v hero-v">{cur["coverage_pct"]:.0f}%</div>
          <div class="s">of full-corpus Vendi</div></div>
      </div>
      <div class="note">Selection maximises log&nbsp;det(I+K) per hour, which is submodular and so
      carries the standard greedy guarantee. Coverage exceeds 100% because dropping redundant
      cycles raises the score above the full corpus &mdash; the discarded half was mostly repetition.
      Most-dropped verb: <b>fold</b> ({cur["dropped_verb_mix"].get("fold", 0)} cycles).</div>
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
</script>"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
