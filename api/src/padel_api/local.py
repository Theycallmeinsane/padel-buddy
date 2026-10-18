"""Padel Buddy on your own computer: the API, a worker and the web app in one process.

    padel-buddy                          start, and open http://127.0.0.1:8000
    padel-buddy import match.mov         add a video already on this computer
    padel-buddy import match.mov --analyze

The web app is the static build of ``web/`` (``npm run build`` writes
``web/out``). It is found next to this checkout, or set ``PADEL_WEB_DIR``.
Data goes to ``PADEL_DATA_DIR`` (default ``./data``).
"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from pathlib import Path

from . import jobs, worker
from .db import Database, Match
from .main import import_video
from .settings import Settings, get_settings

log = logging.getLogger("padel.local")

# api/src/padel_api/local.py -> repo root, for an editable install from a checkout
_CHECKOUT_WEB = Path(__file__).resolve().parents[3] / "web" / "out"


def main(argv: list[str] | None = None, settings: Settings | None = None) -> None:
    parser = argparse.ArgumentParser(prog="padel-buddy", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    imp = sub.add_parser("import", help="Add a video that is already on this computer")
    imp.add_argument("video", type=Path)
    imp.add_argument("--title")
    imp.add_argument("--analyze", action="store_true", help="Queue analysis straight away")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-worker", action="store_true", help="Run workers separately (padel-worker)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = settings or get_settings()
    if args.command == "import":
        _import(settings, args.video, args.title, args.analyze)
    else:
        serve(settings, args.host, args.port, browser=not args.no_browser, with_worker=not args.no_worker)


def _import(settings: Settings, video: Path, title: str | None, analyze: bool) -> None:
    match = import_video(settings, video, title)
    if match.status != "uploaded":
        raise SystemExit(f"Could not import {video}: {match.error}")
    print(f"Added '{match.title}' ({match.duration_s or 0:.0f} s) as match {match.id}")
    if analyze:
        with Database(settings.db_url).sessions() as session:
            jobs.enqueue(session, session.get(Match, match.id))
        print("Analysis queued; it starts when padel-buddy is running.")


def serve(settings: Settings, host: str, port: int, browser: bool = True, with_worker: bool = True) -> None:
    import uvicorn

    from .main import create_app

    if settings.web_dir is None and (_CHECKOUT_WEB / "index.html").exists():
        settings = settings.model_copy(update={"web_dir": _CHECKOUT_WEB})
    if settings.web_dir is None:
        log.warning("No web app build found (run `npm run build` in web/); serving the API only")

    stop = threading.Event()
    if with_worker:
        threading.Thread(target=worker.run_forever, args=(settings, stop), name="worker", daemon=True).start()

    url = f"http://{host}:{port}/"
    config = uvicorn.Config(create_app(settings), host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    if browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"Padel Buddy is running at {url}  (Ctrl+C to stop)")
    try:
        server.run()
    finally:
        stop.set()


if __name__ == "__main__":  # pragma: no cover
    main()
