"""Tiny web UI for browsing EgoVerse kinematic segment manifests.

Deploy it with:
    modal deploy modal_segment_browser.py

Or run a temporary development server with:
    modal serve modal_segment_browser.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal


VOLUME_NAME = "egoverse-zarrs-v2"
MOUNT = Path("/egoverse")
KINEMATIC_SEGMENTS = MOUNT / "kinematic_segments"
EPISODES = MOUNT / "episodes"
VIDEO_SEGMENTS = MOUNT / "segments"
ATTEMPT_SIMILARITY = MOUNT / "attempt_similarity"
ATTEMPTS = MOUNT / "attempts"
ATTEMPT_CLUSTERS = MOUNT / "attempt_clusters"
EPISODE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

app = modal.App("egoverse-segment-browser")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
image = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install(
    "fastapi[standard]==0.116.1",
    "zarr==3.1.5",
    "simplejpeg==1.9.0",
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EgoVerse segment browser</title>
  <style>
    :root{color-scheme:dark;--bg:#0b0d12;--panel:#151922;--muted:#929bad;--line:#292f3c;--accent:#77e2b4}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f3f5f7;font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
    main{max-width:1100px;margin:auto;padding:34px 22px 70px}h1{font-size:26px;margin:0 0 5px}p{color:var(--muted);margin:0 0 24px}
    .controls{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}select,input{background:var(--panel);color:inherit;border:1px solid var(--line);border-radius:8px;padding:9px 11px}
    input{min-width:220px}.summary{display:flex;gap:18px;color:var(--muted);margin-bottom:14px}.summary b{color:#fff}
    .timeline{position:relative;height:46px;margin-bottom:18px;background:#10141b;border-radius:6px;overflow:hidden}.tick{position:absolute;top:0;height:100%;min-width:3px;border:0;border-right:2px solid var(--bg);border-radius:3px;background:#314c48;cursor:pointer}
    .tick:hover,.tick.active{background:var(--accent)}
    .layout{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(300px,.8fr);gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
    video{display:block;width:100%;aspect-ratio:4/3;background:#050608;border-radius:8px}.now{display:flex;justify-content:space-between;gap:10px;margin-top:12px;color:var(--muted)}.now strong{color:var(--accent)}
    h2{font-size:16px;margin:0 0 14px}.details{display:grid;grid-template-columns:1fr 1fr;gap:12px}.details span{display:block;color:var(--muted);font-size:12px}.details strong{font-size:15px}
    .list{margin-top:18px;border-top:1px solid var(--line)}.row{display:grid;grid-template-columns:70px 1fr 100px 90px;gap:12px;padding:10px 4px;border-bottom:1px solid var(--line);cursor:pointer}.row:hover,.row.active{color:var(--accent)}
    .similarity{margin-top:18px}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:#0e1117;border:1px solid var(--line);border-radius:8px;padding:12px}.metric span{display:block;color:var(--muted);font-size:12px}.metric b{font-size:21px}.curve{display:flex;align-items:end;height:100px;gap:8px;margin-top:14px}.bar{flex:1;background:#294b42;border-radius:5px 5px 0 0;min-height:3px;position:relative}.bar span{position:absolute;bottom:-22px;width:100%;text-align:center;color:var(--muted);font-size:11px}.score-row{display:grid;grid-template-columns:1.5fr repeat(5,.7fr);gap:8px;padding:9px 4px;border-bottom:1px solid var(--line)}.score-row.header{color:var(--muted);font-size:11px}.pill{display:inline-block;border-radius:99px;padding:2px 7px;background:#26372f;color:var(--accent);font-size:11px}.note{color:var(--muted);padding:16px 0}
    .evidence{display:flex;height:22px;margin:12px 0 8px;border-radius:4px;overflow:hidden}.window{flex:1;border:0;padding:0}.window.TASK{background:#4fc998}.window.RESET{background:#e89c52}.window.IRRELEVANT{background:#4d5668}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.attempt-info{margin-top:12px;padding:12px;background:#0e1117;border-radius:8px;color:var(--muted)}
    .compare{margin-top:18px}.compare-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}.compare-head select{flex:1;min-width:260px}.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.compare-card video{width:100%}.compare-card h3{font-size:13px;margin:8px 0;color:var(--muted);overflow-wrap:anywhere}.compare-actions{display:flex;gap:8px;margin-top:12px}.compare-actions button{background:var(--accent);color:#07110d;border:0;border-radius:7px;padding:8px 13px;font-weight:700;cursor:pointer}.score-chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.score-chips span{background:#202733;border-radius:99px;padding:4px 8px;color:var(--muted)}
    .empty{padding:40px;text-align:center;color:var(--muted)}
    @media(max-width:750px){.layout,.compare-grid{grid-template-columns:1fr}.row{grid-template-columns:55px 1fr 75px}.row span:nth-child(4){display:none}.metric-grid{grid-template-columns:1fr 1fr}.score-row{grid-template-columns:1fr 1fr}.score-row span:nth-child(n+3){display:none}}
  </style>
</head>
<body><main>
  <h1>EgoVerse segments</h1><p>Play an episode and explore its motion-driven boundaries.</p>
  <div class="controls"><select id="episode"></select><select id="method"><option value="attempts">Attempts</option><option value="kinematic">Kinematic segments</option><option value="fixed_1s">Fixed 1 second</option></select><input id="search" placeholder="Jump to item number…" type="number" min="1"></div>
  <div id="app" class="empty">Loading segment manifests…</div>
</main>
<script>
const state={all:[],data:null,similarity:null,comparisons:[],selected:0};
const fmt=n=>`${Number(n).toFixed(2)}s`;
async function loadEpisodes(){
  const [r,sim,comparisons]=await Promise.all([fetch('/api/episodes'),fetch('/api/similarity'),fetch('/api/comparisons')]); if(!r.ok)throw Error(await r.text()); state.all=await r.json();if(sim.ok)state.similarity=await sim.json();if(comparisons.ok)state.comparisons=await comparisons.json();
  episode.innerHTML=state.all.map(x=>`<option value="${x.episode}">${x.episode} · ${x.attempt_count} attempts${x.has_video?' · ready':''}</option>`).join(''); await load();
}
async function load(){app.className='empty';app.textContent='Loading…';const url=method.value==='attempts'?`/api/attempts/${episode.value}`:`/api/episodes/${episode.value}/${method.value}`;const r=await fetch(url);if(!r.ok){if(method.value==='attempts'){method.value='kinematic';return load()}throw Error(await r.text())}state.data=await r.json();state.selected=0;render()}
function render(){const d=state.data,s=d.segments,total=s.at(-1)?.end_time||0,avg=s.reduce((a,x)=>a+x.end_time-x.start_time,0)/(s.length||1);
  const label=d.method==='attempts'?'attempts':'segments',duration=d.method==='attempts'?(d.windows?.at(-1)?.end_sec||total):total;
  app.className='';app.innerHTML=`<div class="summary"><span><b>${s.length}</b> ${label}</span><span><b>${fmt(duration)}</b> duration</span><span><b>${fmt(avg)}</b> average</span><span><b>${d.fps}</b> fps</span>${d.needs_human_review?'<span class="pill">Needs review</span>':''}</div><div class="timeline">${s.map((x,i)=>`<button class="tick ${i===state.selected?'active':''}" data-i="${i}" style="left:${100*x.start_time/duration}%;width:${100*(x.end_time-x.start_time)/duration}%" title="#${i+1} · ${fmt(x.start_time)}–${fmt(x.end_time)}"></button>`).join('')}</div><div class="layout"><section class="panel"><video id="player" controls preload="metadata" src="/api/video/${d.episode}"></video><div class="now"><span>Current: <strong id="currentSegment">${d.method==='attempts'?'Outside attempt':'Segment 1'}</strong></span><span id="currentTime">0.00s / ${fmt(duration)}</span></div>${d.method==='attempts'?renderEvidence(d):''}</section><section class="panel"><h2>All ${label}</h2><div class="list">${s.map((x,i)=>`<div class="row ${i===state.selected?'active':''}" data-i="${i}"><span>#${x.attempt_id??i+1}</span><span>${fmt(x.start_time)}–${fmt(x.end_time)}</span><span>${x.end_idx-x.start_idx} frames</span><span>${x.confidence==null?fmt(x.end_time-x.start_time):pct(x.confidence)}</span></div>`).join('')}</div></section></div><section class="panel compare" id="comparison"></section><section class="panel similarity" id="similarity"></section>`;
  document.querySelectorAll('[data-i]').forEach(el=>el.onclick=()=>seekSegment(+el.dataset.i));document.querySelectorAll('[data-time]').forEach(el=>el.onclick=()=>{player.currentTime=+el.dataset.time});player.ontimeupdate=syncSegment;renderComparison();renderSimilarity();
}
function seekSegment(i){state.selected=Math.max(0,Math.min(i,state.data.segments.length-1));player.currentTime=state.data.segments[state.selected].start_time;syncSegment()}
function syncSegment(){const t=player.currentTime,s=state.data.segments,found=s.findIndex(x=>t>=x.start_time&&t<x.end_time),i=found<0&&state.data.method!=='attempts'&&s.length&&t>=s.at(-1).end_time?s.length-1:found;if(i!==state.selected){state.selected=i;document.querySelectorAll('[data-i]').forEach(x=>x.classList.toggle('active',+x.dataset.i===i))}currentSegment.textContent=i<0?'Outside attempt':`${state.data.method==='attempts'?'Attempt':'Segment'} ${state.data.method==='attempts'?(s[i].attempt_id??i+1):i+1}`;currentTime.textContent=`${fmt(t)} / ${fmt(state.data.windows?.at(-1)?.end_sec||s.at(-1)?.end_time||0)}`;if(state.data.method==='attempts'){const w=state.data.windows?.find(x=>t>=x.start_sec&&t<x.end_sec);windowDetail.textContent=w?`${w.final_class} · ${pct(w.confidence)} confidence · ${w.visual_reason}`:'No window evidence'}}
const pct=n=>`${Math.round(100*n)}%`;
function renderEvidence(d){return `<div class="evidence">${(d.windows||[]).map(x=>`<button class="window ${x.final_class}" data-time="${x.start_sec}" title="${x.final_class}: ${x.visual_reason}"></button>`).join('')}</div><div class="legend"><span><i class="dot" style="background:#4fc998"></i>Task</span><span><i class="dot" style="background:#e89c52"></i>Reset</span><span><i class="dot" style="background:#4d5668"></i>Irrelevant</span></div><div class="attempt-info" id="windowDetail">Move through the video to inspect window evidence.</div>`}
function renderComparison(){const rows=state.comparisons;if(!rows.length){comparison.innerHTML='<h2>Compare similar attempts</h2><div class="note">No comparable attempt pairs found.</div>';return}comparison.innerHTML=`<div class="compare-head"><h2>Compare similar attempts</h2><select id="pairSelect">${rows.map((x,i)=>`<option value="${i}">${pct(x.overall_similarity)} · ${x.attempt_a} ↔ ${x.attempt_b}</option>`).join('')}</select></div><div id="pairView"></div>`;pairSelect.onchange=showPair;showPair()}
function showPair(){const x=state.comparisons[+pairSelect.value];pairView.innerHTML=`<div class="compare-grid"><div class="compare-card"><video id="leftVideo" controls preload="metadata" src="/api/video/${x.left.episode}"></video><h3>${x.attempt_a} · ${fmt(x.left.start_time)}–${fmt(x.left.end_time)}</h3></div><div class="compare-card"><video id="rightVideo" controls preload="metadata" src="/api/video/${x.right.episode}"></video><h3>${x.attempt_b} · ${fmt(x.right.start_time)}–${fmt(x.right.end_time)}</h3></div></div><div class="score-chips"><span>Overall ${pct(x.overall_similarity)}</span><span>Trajectory ${pct(x.trajectory_similarity)}</span><span>Orientation ${pct(x.orientation_similarity)}</span><span>Coordination ${pct(x.coordination_similarity)}</span><span>Dynamics ${pct(x.dynamics_similarity)}</span></div><div class="compare-actions"><button id="playBoth">Play both</button><button id="pauseBoth">Pause</button><button id="restartBoth">Restart attempts</button></div>`;const l=leftVideo,r=rightVideo;const seek=()=>{l.currentTime=x.left.start_time;r.currentTime=x.right.start_time};let ready=0;[l,r].forEach(v=>v.onloadedmetadata=()=>{if(++ready===2)seek()});playBoth.onclick=async()=>{if(l.currentTime>=x.left.end_time||r.currentTime>=x.right.end_time)seek();await Promise.all([l.play(),r.play()])};pauseBoth.onclick=()=>{l.pause();r.pause()};restartBoth.onclick=()=>{l.pause();r.pause();seek()};l.ontimeupdate=()=>{if(l.currentTime>=x.left.end_time){l.pause();r.pause()}};r.ontimeupdate=()=>{if(r.currentTime>=x.right.end_time){l.pause();r.pause()}}}
function renderSimilarity(){
  const payload=state.similarity;
  if(!payload){similarity.innerHTML='<h2>Attempt similarity</h2><div class="note">No similarity artifacts found.</div>';return}
  const sum=payload.summary,task=payload.tasks[0],pairs=task?.pairwise||[],curve=task?.threshold_curve||sum.threshold_curve||[],cluster=payload.clusters?.tasks?.[0];
  similarity.innerHTML=`<h2>Attempt similarity · ${task?.task_id||'dataset'}</h2><div class="metric-grid"><div class="metric"><span>Attempts</span><b>${sum.attempts_originally}</b></div><div class="metric"><span>Kept</span><b>${sum.attempts_kept}</b></div><div class="metric"><span>Mean coverage</span><b>${pct(sum.mean_behavioral_coverage)}</b></div><div class="metric"><span>Redundancy threshold</span><b>${pct(sum.similarity_config.redundancy_threshold)}</b></div></div><h2 style="margin-top:22px">Retention by threshold</h2><div class="curve">${curve.map(x=>`<div class="bar" style="height:${Math.max(3,100*x.retained_fraction)}%" title="${pct(x.retained_fraction)} retained"><span>${x.threshold}</span></div>`).join('')}</div><h2 style="margin-top:34px">Pairwise scores</h2>${pairs.length?`<div class="score-row header"><span>Attempt pair</span><span>Overall</span><span>Trajectory</span><span>Orientation</span><span>Coordination</span><span>Dynamics</span></div>${pairs.map(x=>`<div class="score-row"><span>${x.attempt_a}<br>${x.attempt_b}</span><b>${pct(x.overall_similarity)}</b><span>${pct(x.trajectory_similarity)}</span><span>${pct(x.orientation_similarity)}</span><span>${pct(x.coordination_similarity)}</span><span>${pct(x.dynamics_similarity)}</span></div>`).join('')}`:`<div class="note">Only one attempt has been processed, so there are no attempt pairs to compare yet.</div>`}<h2 style="margin-top:20px">Curation</h2>${(task?.curation||[]).map(x=>`<div class="score-row"><span>${x.attempt_id}</span><span><span class="pill">${x.decision}</span></span><span>${pct(x.similarity)} coverage</span></div>`).join('')}${cluster?`<h2 style="margin-top:20px">Clusters</h2><div class="metric-grid"><div class="metric"><span>Clusters</span><b>${cluster.metrics.clusters}</b></div><div class="metric"><span>Attempts dropped</span><b>${cluster.metrics.attempts_dropped}</b></div><div class="metric"><span>Nearest-kept coverage</span><b>${pct(cluster.metrics.mean_nearest_kept_coverage)}</b></div><div class="metric"><span>Threshold</span><b>${pct(cluster.clustering.similarity_threshold)}</b></div></div>${cluster.clusters.map(x=>`<div class="attempt-info"><b>Cluster ${x.cluster_id}</b> · ${x.size} attempt${x.size===1?'':'s'} · medoid ${x.medoid_attempt_id}</div>`).join('')}`:''}`;
}
episode.onchange=()=>{method.value='attempts';load()};method.onchange=load;search.onchange=()=>seekSegment(Number(search.value)-1);loadEpisodes().catch(e=>{app.textContent=e.message});
</script></body></html>"""


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import subprocess
    import threading
    import zarr
    import simplejpeg
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    api = FastAPI()
    volume.reload()
    video_lock = threading.Lock()

    def safe_episode(episode: str) -> str:
        if not EPISODE_ID.fullmatch(episode):
            raise HTTPException(400, "Invalid episode id")
        return episode

    @api.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @api.get("/api/episodes")
    def episodes():
        results = []
        if ATTEMPTS.is_dir():
            for manifest in sorted(ATTEMPTS.glob("*/attempts.json")):
                data = json.loads(manifest.read_text())
                episode = manifest.parent.name
                results.append(
                    {
                        "episode": episode,
                        "attempt_count": len(data.get("attempts", [])),
                        "has_video": (VIDEO_SEGMENTS / episode / "front_1").is_dir(),
                        "has_kinematic": (KINEMATIC_SEGMENTS / episode / "kinematic.json").is_file(),
                        "has_features": (MOUNT / "attempt_features" / episode / "features.json").is_file(),
                    }
                )
        return results

    @api.get("/api/episodes/{episode}/{method}")
    def manifest(episode: str, method: str):
        safe_episode(episode)
        if method not in {"kinematic", "fixed_1s"}:
            raise HTTPException(400, "Unknown segmentation method")
        path = KINEMATIC_SEGMENTS / episode / f"{method}.json"
        if not path.is_file():
            raise HTTPException(404, "Manifest not found")
        return json.loads(path.read_text())

    @api.get("/api/attempts/{episode}")
    def attempts(episode: str):
        safe_episode(episode)
        path = ATTEMPTS / episode / "attempts.json"
        if not path.is_file():
            raise HTTPException(404, "Attempt manifest not found")
        payload = json.loads(path.read_text())
        fps = 30.0
        segment_path = KINEMATIC_SEGMENTS / episode / "kinematic.json"
        if segment_path.is_file():
            fps = float(json.loads(segment_path.read_text()).get("fps", fps))
        payload["episode"] = episode
        payload["method"] = "attempts"
        payload["fps"] = fps
        payload["segments"] = [
            {
                "start_idx": item["start_idx"],
                "end_idx": item["end_idx"],
                "start_time": item["start_sec"],
                "end_time": item["end_sec"],
                "confidence": item.get("confidence"),
                "attempt_id": item["attempt_id"],
            }
            for item in payload.get("attempts", [])
        ]
        return payload

    @api.get("/api/similarity")
    def similarity():
        summary_path = ATTEMPT_SIMILARITY / "summary.json"
        if not summary_path.is_file():
            raise HTTPException(404, "Similarity summary not found")
        summary = json.loads(summary_path.read_text())
        tasks = []
        for item in summary.get("tasks", []):
            task_path = ATTEMPT_SIMILARITY / "tasks" / Path(item["result_path"]).name
            if task_path.is_file():
                task = json.loads(task_path.read_text())
                task["pairwise"] = sorted(
                    task.get("pairwise", []), key=lambda row: row["overall_similarity"], reverse=True
                )[:100]
                tasks.append(task)
        clusters = None
        cluster_summary = ATTEMPT_CLUSTERS / "summary.json"
        if cluster_summary.is_file():
            clusters = {"summary": json.loads(cluster_summary.read_text()), "tasks": []}
            for path in sorted((ATTEMPT_CLUSTERS / "tasks").glob("*.json")):
                task = json.loads(path.read_text())
                task.pop("visualization", None)
                for attempt in task.get("attempts", []):
                    attempt.pop("similar_attempts", None)
                clusters["tasks"].append(task)
        return {"summary": summary, "tasks": tasks, "clusters": clusters}

    @api.get("/api/comparisons")
    def comparisons(limit: int = 100):
        limit = min(250, max(1, limit))
        bounds = {}
        for path in sorted(ATTEMPTS.glob("*/attempts.json")):
            payload = json.loads(path.read_text())
            episode = path.parent.name
            for item in payload.get("attempts", []):
                bounds[f"{episode}:{int(item['attempt_id'])}"] = {
                    "episode": episode,
                    "start_time": float(item["start_sec"]),
                    "end_time": float(item["end_sec"]),
                }
        pairs = []
        for path in sorted((ATTEMPT_SIMILARITY / "tasks").glob("*.json")):
            for pair in json.loads(path.read_text()).get("pairwise", []):
                if pair["attempt_a"] in bounds and pair["attempt_b"] in bounds:
                    pairs.append({**pair, "left": bounds[pair["attempt_a"]], "right": bounds[pair["attempt_b"]]})
        pairs.sort(key=lambda row: row["overall_similarity"], reverse=True)
        return pairs[:limit]

    @api.get("/api/video/{episode}")
    def video(episode: str):
        safe_episode(episode)
        source = VIDEO_SEGMENTS / episode / "front_1"
        clips = sorted(source.glob("*.mp4"))
        output = Path("/tmp") / f"{episode}.mp4"
        with video_lock:
            if not output.is_file():
                if clips:
                    concat_file = Path("/tmp") / f"{episode}-concat.txt"
                    concat_file.write_text("".join(f"file '{clip}'\n" for clip in clips))
                    result = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", "-y", str(output)], capture_output=True, text=True
                    )
                    if result.returncode:
                        raise HTTPException(500, f"Could not assemble video: {result.stderr}")
                else:
                    store_path = EPISODES / episode
                    if not store_path.is_dir():
                        raise HTTPException(404, "Episode not found")
                    store = zarr.open_group(str(store_path), mode="r")
                    frames = store["images.front_1"]
                    total_frames = min(int(store.attrs.get("total_frames", frames.shape[0])), int(frames.shape[0]))
                    fps = float(store.attrs.get("fps", 30.0))
                    first_payload = frames[0:1][0]
                    if hasattr(first_payload, "item"):
                        first_payload = first_payload.item()
                    first = simplejpeg.decode_jpeg(bytes(first_payload), colorspace="RGB")
                    height, width = first.shape[:2]
                    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(output)]
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    assert process.stdin is not None
                    process.stdin.write(first.tobytes())
                    for index in range(1, total_frames):
                        payload = frames[index : index + 1][0]
                        if hasattr(payload, "item"):
                            payload = payload.item()
                        process.stdin.write(simplejpeg.decode_jpeg(bytes(payload), colorspace="RGB").tobytes())
                    process.stdin.close()
                    stderr = process.stderr.read() if process.stderr else b""
                    if process.wait():
                        output.unlink(missing_ok=True)
                        raise HTTPException(500, f"Could not encode video: {stderr.decode(errors='replace')}")
        return FileResponse(output, media_type="video/mp4", headers={"Cache-Control": "public, max-age=3600"})

    return api
