#!/usr/bin/env python3
"""Serve the isolated EgoTrim V2 dashboard using only the Python standard library."""

from __future__ import annotations

import argparse
import functools
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote


EXPERIMENT_ROOT = Path(__file__).resolve().parent


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Downloaded run directory; defaults to the newest directory under live_runs",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.run_dir is None:
        candidates = [path for path in (EXPERIMENT_ROOT / "live_runs").glob("*") if path.is_dir()]
        if not candidates:
            raise SystemExit("No downloaded runs found under live_runs; pass --run-dir")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    else:
        run_dir = args.run_dir
        if not run_dir.is_absolute():
            run_dir = (EXPERIMENT_ROOT / run_dir).resolve()
    run_dir = run_dir.resolve(strict=True)
    try:
        relative_run = run_dir.relative_to(EXPERIMENT_ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"Run directory must be inside {EXPERIMENT_ROOT}: {run_dir}") from exc

    required = ("metrics.json", "dashboard_data.json", "modal_run_manifest.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Run directory is missing: {', '.join(missing)}")
    if not (EXPERIMENT_ROOT / "dashboard.html").is_file():
        raise SystemExit("Missing dashboard.html")

    handler = functools.partial(QuietHandler, directory=str(EXPERIMENT_ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = (
        f"http://{args.host}:{server.server_port}/dashboard.html"
        f"?run={quote(relative_run)}"
    )
    print(f"EgoTrim V2 dashboard: {url}")
    print(f"Run directory: {run_dir}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
