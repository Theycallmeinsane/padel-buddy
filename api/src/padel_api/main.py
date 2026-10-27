"""HTTP API used by the web app (and later the Android and desktop apps).

Flow for one match:

    POST /matches                         create, get an id
    PUT  /matches/{id}/video              upload, in resumable chunks (Content-Range)
    GET  /matches/{id}/frame?t=5          a still frame to click the court on
    PUT  /matches/{id}/calibration        the clicked points (optional: enables tracking)
    POST /matches/{id}/analyze            queue the engine
    GET  /matches/{id}                    status and progress, and how far live results reach
    GET  /matches/{id}/results            tracking, clips, ball, shots and report (so far, while analysing)
    GET  /matches/{id}/files/{path}       any output: clips, reel, heatmaps, overlay data
    PUT  /matches/{id}/players            name the players: {"names": {"P1": "Maya"}}
    POST /matches/{id}/tags               name the person at a spot at a moment (fixes mix-ups too)
    DELETE /matches/{id}/tags/{tag}       take a tag back
    GET  /matches/{id}/players/{P1..P4}   one player: numbers, shots, highlights, reel
    GET  /matches/{id}/report.md          the match report, with names
    GET  /people                          everyone named in any match
    GET  /people/{name}                   one person across all their matches

Run with ``padel-api`` (or ``uvicorn padel_api.main:create_app --factory``).
``padel-buddy`` (see ``local.py``) runs the API, a worker and the web app in
one process for use on your own computer.
"""

from __future__ import annotations

import gzip
import json
import re
import secrets
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import __version__, jobs
from .db import Database, Job, Match, new_id, now
from .schemas import (
    AnalyzeIn, CalibrationIn, CalibrationOut, JobOut, Keypoint, LiveOut, MatchCreate, MatchOut, PersonOut,
    PlayerNamesIn, PlayerOut, TagIn, TagOut, UploadStatus,
)
from .settings import Settings, get_settings
from .storage import ALLOWED_VIDEO_EXTS, LocalStorage

_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
PLAYER_IDS = ("P1", "P2", "P3", "P4")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_url)
    db.create_all()
    storage = LocalStorage(settings.data_dir)

    app = FastAPI(title="Padel Buddy API", version=__version__)
    app.state.db = db
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    def get_session() -> Iterator[Session]:
        yield from db.session()

    # Closed when the endpoint returns, not once the response is sent: a browser
    # streams a video or clip for as long as it plays, and a connection held for
    # each stream ran the pool dry.
    session_dep = Depends(get_session, scope="function")

    def require_token(
        authorization: str | None = Header(None), token: str | None = Query(None)
    ) -> None:
        if not settings.api_token:
            return
        given = token or (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given, settings.api_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or wrong API token")

    def load_match(match_id: str, session: Session = session_dep) -> Match:
        match = session.get(Match, match_id)
        if match is None:
            raise HTTPException(404, "Match not found")
        return match

    auth = [Depends(require_token)]

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/court/keypoints", dependencies=auth)
    def keypoints() -> list[Keypoint]:
        from padel_engine import court

        return [Keypoint(name=n, x_m=x, y_m=y) for n, (x, y) in court.KEYPOINTS.items()]

    def match_out(match: Match) -> MatchOut:
        return _match_out(match, storage.run_dir(match.id))

    # -- matches ----------------------------------------------------------

    @app.post("/matches", status_code=201, dependencies=auth)
    def create_match(body: MatchCreate, session: Session = session_dep) -> MatchOut:
        ext = Path(body.filename).suffix.lower() or ".mp4"
        if ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(415, f"Unsupported video type {ext}; use one of {sorted(ALLOWED_VIDEO_EXTS)}")
        _check_size(body.size, settings)
        match = Match(title=body.title, video_ext=ext, video_size=body.size)
        session.add(match)
        session.commit()
        storage.match_dir(match.id).mkdir(parents=True, exist_ok=True)
        return match_out(match)

    @app.get("/matches", dependencies=auth)
    def list_matches(session: Session = session_dep) -> list[MatchOut]:
        rows = session.scalars(select(Match).order_by(Match.created_at.desc())).all()
        return [match_out(m) for m in rows]

    @app.get("/matches/{match_id}", dependencies=auth)
    def get_match(match: Match = Depends(load_match)) -> MatchOut:
        return match_out(match)

    @app.delete("/matches/{match_id}", status_code=204, dependencies=auth)
    def delete_match(match: Match = Depends(load_match), session: Session = session_dep) -> Response:
        job = jobs.active_job(session, match.id)
        if job is not None and job.status == "running":
            raise HTTPException(409, "The match is being analysed; delete it when the job finishes")
        session.delete(session.merge(match))
        session.commit()
        storage.delete(match.id)
        return Response(status_code=204)

    # -- upload -----------------------------------------------------------

    @app.get("/matches/{match_id}/video/status", dependencies=auth)
    def upload_status(match: Match = Depends(load_match)) -> UploadStatus:
        return _upload_status(match)

    @app.put("/matches/{match_id}/video", dependencies=auth)
    async def upload_video(
        request: Request,
        match: Match = Depends(load_match),
        session: Session = session_dep,
        content_range: str | None = Header(None),
    ) -> UploadStatus:
        """Upload the whole file, or one chunk with ``Content-Range: bytes a-b/total``.

        Chunks must start at or before the bytes already received, so after a
        dropped connection the client asks /video/status and carries on from
        ``received``.
        """
        match = session.merge(match)
        if match.status not in ("created", "uploading"):
            raise HTTPException(409, f"Upload already finished (status {match.status})")

        start, end, total = 0, None, None
        if content_range:
            m = _RANGE.fullmatch(content_range.strip())
            if not m:
                raise HTTPException(400, "Content-Range must look like 'bytes 0-1048575/52428800'")
            start, end = int(m[1]), int(m[2])
            total = None if m[3] == "*" else int(m[3])
        if total is not None:
            _check_size(total, settings)
            match.video_size = total
        if start > match.video_received:
            raise HTTPException(
                409, f"Chunk starts at {start} but only {match.video_received} bytes are stored; resume from there"
            )

        path = storage.video_path(match.id, match.video_ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        limit = int(settings.max_upload_gb * 1024**3)
        written = 0
        with open(path, "r+b" if path.exists() else "wb") as fh:
            fh.seek(start)
            async for chunk in request.stream():
                if start + written + len(chunk) > limit:
                    raise HTTPException(413, "Upload exceeds the size limit")
                fh.write(chunk)
                written += len(chunk)
            if not content_range:
                fh.truncate()
        if end is not None and written != end - start + 1:
            raise HTTPException(400, f"Expected {end - start + 1} bytes in this chunk, got {written}")

        match.video_received = max(match.video_received, start + written)
        if not content_range:
            match.video_size = match.video_received
        match.status = "uploading"
        if match.video_size is not None and match.video_received >= match.video_size:
            _finish_upload(match, path)
        session.commit()
        return _upload_status(match)

    @app.get("/matches/{match_id}/video", dependencies=auth)
    def get_video(match: Match = Depends(load_match)) -> FileResponse:
        path = storage.video_path(match.id, match.video_ext)
        if not _uploaded(match) or not path.exists():
            raise HTTPException(404, "Video not uploaded yet")
        return FileResponse(path)

    # -- calibration ------------------------------------------------------

    @app.get("/matches/{match_id}/frame", dependencies=auth)
    def frame(match: Match = Depends(load_match), t: float = Query(5.0, ge=0)) -> Response:
        import cv2
        from padel_engine.video import read_frame

        if not _uploaded(match):
            raise HTTPException(409, "Upload the video first")
        try:
            img = read_frame(storage.video_path(match.id, match.video_ext), t)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={
            "X-Frame-Width": str(img.shape[1]), "X-Frame-Height": str(img.shape[0]),
        })

    @app.put("/matches/{match_id}/calibration", dependencies=auth)
    def set_calibration(
        body: CalibrationIn, match: Match = Depends(load_match), session: Session = session_dep
    ) -> CalibrationOut:
        from padel_engine.calibration import Calibration, CalibrationError

        try:
            calib = Calibration.from_points(dict(body.points), body.image_size)
        except CalibrationError as exc:
            raise HTTPException(422, str(exc)) from exc
        match = session.merge(match)
        match.calibration = calib.to_dict()
        session.commit()
        return CalibrationOut(points_used=sorted(calib.points), reprojection_error_m=calib.reprojection_error_m)

    @app.delete("/matches/{match_id}/calibration", status_code=204, dependencies=auth)
    def clear_calibration(match: Match = Depends(load_match), session: Session = session_dep) -> Response:
        match = session.merge(match)
        match.calibration = None
        session.commit()
        return Response(status_code=204)

    # -- analysis and results ---------------------------------------------

    @app.post("/matches/{match_id}/analyze", status_code=202, dependencies=auth)
    def analyze(
        body: AnalyzeIn | None = None, match: Match = Depends(load_match), session: Session = session_dep
    ) -> JobOut:
        from padel_engine.config import EngineConfig

        match = session.merge(match)
        if not _uploaded(match):
            raise HTTPException(409, "Upload the video first")
        if jobs.active_job(session, match.id):
            raise HTTPException(409, "This match is already queued or being analysed")
        config = body.config if body else None
        if config:
            try:
                EngineConfig.model_validate(config)
            except ValueError as exc:
                raise HTTPException(422, f"Invalid engine config: {exc}") from exc
        return _job_out(jobs.enqueue(session, match, config))

    @app.get("/matches/{match_id}/results", dependencies=auth)
    def results(match: Match = Depends(load_match)) -> dict:
        run = storage.run_dir(match.id)
        out = {}
        for key, name in (
            ("tracking", "summary.json"), ("clips", "clips.json"), ("ball", "ball.json"),
            ("shots", "shots.json"), ("report", "report.json"),
        ):
            f = run / name
            out[key] = json.loads(f.read_bytes()) if f.exists() else None
        return out

    # -- players and people -------------------------------------------------

    @app.put("/matches/{match_id}/players", dependencies=auth)
    def name_players(
        body: PlayerNamesIn, match: Match = Depends(load_match), session: Session = session_dep
    ) -> MatchOut:
        unknown = set(body.names) - set(PLAYER_IDS)
        if unknown:
            raise HTTPException(422, f"Players are {', '.join(PLAYER_IDS)}; got {', '.join(sorted(unknown))}")
        match = session.merge(match)
        names = dict(match.player_names or {})
        tags = list(match.tags or [])
        forgot = False
        for pid, name in body.names.items():
            name = _clean_name(name)
            old = names.get(pid)
            if old and old != name:
                # Tags carry the name: a renamed player keeps their tags, a cleared one loses them.
                same = [t for t in tags if _same_name(t["name"], old)]
                forgot |= not name and any(t.get("applied") for t in same)
                tags = [{**t, "name": name} if t in same else t for t in tags if name or t not in same]
            if name:
                names[pid] = name
            else:
                names.pop(pid, None)
        match.player_names = names or None
        match.tags = tags or None
        session.commit()
        if forgot:
            queue_relabel(session, match)
        return match_out(match)

    @app.post("/matches/{match_id}/tags", status_code=201, dependencies=auth)
    def add_tag(body: TagIn, match: Match = Depends(load_match), session: Session = session_dep) -> MatchOut:
        """Name the person at a spot in the video at one moment.

        Before analysis the tag waits for it. On a player's box in an
        analysed match the name shows at once, unless an earlier tag gave
        that name (or this player's name) to someone else: then tracking
        mixed people up, and who is who is decided again in the background,
        from the last time that player passed close to someone. A tag on a
        frame rather than a box is placed the same way, in the background.
        """
        if body.player is not None and body.player not in PLAYER_IDS:
            raise HTTPException(422, f"Players are {', '.join(PLAYER_IDS)}")
        name = _clean_name(body.name)
        if not name:
            raise HTTPException(422, "Give the player a name")
        match = session.merge(match)
        tags = list(match.tags or [])
        names = dict(match.player_names or {})
        name = next((n for n in [*names.values(), *(t["name"] for t in tags)] if _same_name(n, name)), name)
        tagged = {t["name"].casefold() for t in tags}
        holder = next((pid for pid, n in names.items() if _same_name(n, name)), None)
        current = names.get(body.player) if body.player else None
        shown = body.player is not None and match.status == "done"
        mixed_up = shown and (
            (holder not in (None, body.player) and name.casefold() in tagged)
            or (current is not None and not _same_name(current, name) and current.casefold() in tagged)
        )
        if shown and not mixed_up:
            if holder is not None:
                names.pop(holder)  # a name given without a tag moves to whoever was tagged
            names[body.player] = name
        everyone = {n.casefold() for n in names.values()} | tagged | {name.casefold()}
        if len(everyone) > len(PLAYER_IDS):
            raise HTTPException(
                409, "This match already has four names. Tag one of them, or rename a player in the report first."
            )
        tag = {
            "id": new_id()[:12], "time_s": round(body.time_s, 3), "x": round(body.x, 1), "y": round(body.y, 1),
            "name": name, "player": body.player, "applied": True if shown and not mixed_up else None,
            "created_at": now().isoformat(),
        }
        match.tags = [*tags, tag]
        match.player_names = names or None
        session.commit()
        if mixed_up or not shown or jobs.active_job(session, match.id) is not None:
            # Who is who needs deciding again, or a running job started without this tag.
            queue_relabel(session, match)
        return match_out(match)

    @app.delete("/matches/{match_id}/tags/{tag_id}", dependencies=auth)
    def delete_tag(tag_id: str, match: Match = Depends(load_match), session: Session = session_dep) -> MatchOut:
        match = session.merge(match)
        tags = list(match.tags or [])
        gone = next((t for t in tags if t["id"] == tag_id), None)
        if gone is None:
            raise HTTPException(404, "No such tag")
        match.tags = [t for t in tags if t is not gone] or None
        session.commit()
        if gone.get("applied"):
            queue_relabel(session, match)
        return match_out(match)

    def queue_relabel(session: Session, match: Match) -> None:
        """Decide who is who again from the tags, after whatever is already queued or running.

        A queued job reads the latest tags when it starts, so one is enough.
        A run from before tags existed has no record of its track pieces:
        its tags wait for the next analysis.
        """
        running = any(j.status == "running" for j in match.jobs)
        pieces = (storage.run_dir(match.id) / "segments.json").exists()
        if match.calibration is None or any(j.status == "queued" for j in match.jobs):
            return
        if not running and not (match.status == "done" and pieces):
            return
        config = next((j.config for j in reversed(match.jobs) if j.kind == "analyze"), None)
        jobs.enqueue(session, match, config, kind="relabel")

    @app.get("/matches/{match_id}/players/{player}", dependencies=auth)
    def player(player: str, match: Match = Depends(load_match)) -> PlayerOut:
        if player not in PLAYER_IDS:
            raise HTTPException(404, f"Players are {', '.join(PLAYER_IDS)}")
        return _player_out(match, player, storage.run_dir(match.id))

    @app.get("/matches/{match_id}/report.md", dependencies=auth)
    def report_markdown(match: Match = Depends(load_match)) -> Response:
        from padel_engine.report import to_markdown
        from padel_engine.schemas import MatchReport

        f = storage.run_dir(match.id) / "report.json"
        if not f.exists():
            raise HTTPException(404, "No report yet: analyse the match with a calibration")
        report = MatchReport.model_validate_json(f.read_bytes())
        return Response(to_markdown(report, match.player_names or {}), media_type="text/markdown; charset=utf-8")

    @app.get("/people", dependencies=auth)
    def people(session: Session = session_dep) -> list[PersonOut]:
        counts: dict[str, list] = {}
        for match in session.scalars(select(Match)):
            for name in (match.player_names or {}).values():
                entry = counts.setdefault(name.casefold(), [name, 0])
                entry[1] += 1
        return [PersonOut(name=n, matches=c) for n, c in sorted(counts.values(), key=lambda e: e[0].casefold())]

    @app.get("/people/{name}", dependencies=auth)
    def person(name: str, session: Session = session_dep) -> list[PlayerOut]:
        """Everything about one person: each match they were named in, newest first."""
        key = " ".join(name.split()).casefold()
        found = []
        for match in session.scalars(select(Match).order_by(Match.created_at.desc())):
            for pid, given in sorted((match.player_names or {}).items()):
                if given.casefold() == key:
                    found.append(_player_out(match, pid, storage.run_dir(match.id)))
        if not found:
            raise HTTPException(404, f"Nobody called {name!r} in any match")
        return found

    @app.get("/matches/{match_id}/files/{rel:path}", dependencies=auth)
    def run_file(rel: str, request: Request, match: Match = Depends(load_match)) -> Response:
        path = storage.run_file(match.id, rel)
        if path is None:
            raise HTTPException(404, "No such output")
        if path.suffix == ".json" and "gzip" in request.headers.get("accept-encoding", ""):
            # A long match's overlay is megabytes of numbers; gzip makes it about 5x smaller.
            return Response(
                gzip.compress(path.read_bytes(), compresslevel=5),
                media_type="application/json",
                headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
            )
        return FileResponse(path)

    if settings.web_dir is not None:
        # Mounted last, so API routes win; html=True serves /match/ as match/index.html.
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    return app


def import_video(settings: Settings, src: Path, title: str | None = None) -> Match:
    """Register a video already on this computer as a match, without uploading it.

    The file is hard-linked into the data folder when both are on the same
    drive (instant, no extra space; deleting the match leaves the original),
    and copied otherwise.
    """
    import os
    import shutil

    src = Path(src)
    ext = src.suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise ValueError(f"Unsupported video type {ext}; use one of {sorted(ALLOWED_VIDEO_EXTS)}")
    if not src.is_file():
        raise FileNotFoundError(src)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_url)
    db.create_all()
    storage = LocalStorage(settings.data_dir)
    with db.sessions() as session:
        size = src.stat().st_size
        match = Match(title=title or src.stem, video_ext=ext, video_size=size, status="uploading")
        session.add(match)
        session.commit()
        dest = storage.video_path(match.id, ext)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        match.video_received = size
        _finish_upload(match, dest)
        session.commit()
        session.refresh(match)
        session.expunge(match)
        return match


# -- helpers ------------------------------------------------------------------


def _check_size(size: int | None, settings: Settings) -> None:
    if size is not None and size > settings.max_upload_gb * 1024**3:
        raise HTTPException(413, f"Videos up to {settings.max_upload_gb:g} GB are accepted")


def _uploaded(match: Match) -> bool:
    return match.status not in ("created", "uploading")


def _finish_upload(match: Match, path: Path) -> None:
    from padel_engine.video import probe

    match.status = "uploaded"
    try:
        match.duration_s = probe(path).duration_s
    except FileNotFoundError:
        match.status = "failed"
        match.error = "The uploaded file could not be opened as a video"


def _upload_status(match: Match) -> UploadStatus:
    return UploadStatus(received=match.video_received, total=match.video_size, complete=_uploaded(match))


def _clean_name(name: str) -> str:
    return " ".join(name.split())[:60]


def _same_name(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id, kind=job.kind, status=job.status, progress=job.progress, error=job.error,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
    )


def _match_out(match: Match, run: Path | None = None) -> MatchOut:
    latest = match.jobs[-1] if match.jobs else None
    return MatchOut(
        id=match.id, title=match.title, created_at=match.created_at, status=match.status,
        video_size=match.video_size, video_received=match.video_received, duration_s=match.duration_s,
        calibrated=match.calibration is not None, error=match.error, player_names=match.player_names or {},
        tags=[TagOut.model_validate(t) for t in match.tags or []],
        job=_job_out(latest) if latest else None,
        live=_live(run) if run is not None and match.status != "done" else None,
    )


def _live(run: Path) -> LiveOut | None:
    """How far the results published during an analysis reach; None once it finished."""
    f = run / "live.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_bytes())
    except (OSError, ValueError):
        return None
    return None if data.get("done") else LiveOut(covered_s=data["covered_s"], duration_s=data["duration_s"])


def _player_out(match: Match, player: str, run: Path) -> PlayerOut:
    report = json.loads((run / "report.json").read_bytes()) if (run / "report.json").exists() else None
    shots = json.loads((run / "shots.json").read_bytes())["shots"] if (run / "shots.json").exists() else []
    mine = next((p for p in (report or {}).get("players", []) if p["player"] == player), None)
    return PlayerOut(
        match_id=match.id,
        match_title=match.title,
        match_date=match.created_at,
        player=player,
        name=(match.player_names or {}).get(player),
        team=mine["team"] if mine else None,
        report=mine,
        shots=[s for s in shots if s["player"] == player],
        reel=mine.get("reel") if mine else None,
        winners_reel=mine.get("winners_reel") if mine else None,
        shorts=[x for x in (report or {}).get("shorts", []) if x.get("player") == player],
    )


def serve() -> None:  # pragma: no cover - entry point
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Padel Buddy API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("padel_api.main:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
