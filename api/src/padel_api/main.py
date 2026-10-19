"""HTTP API used by the web app (and later the Android and desktop apps).

Flow for one match:

    POST /matches                         create, get an id
    PUT  /matches/{id}/video              upload, in resumable chunks (Content-Range)
    GET  /matches/{id}/frame?t=5          a still frame to click the court on
    PUT  /matches/{id}/calibration        the clicked points (optional: enables tracking)
    POST /matches/{id}/analyze            queue the engine
    GET  /matches/{id}                    status and progress
    GET  /matches/{id}/results            tracking + clips summaries
    GET  /matches/{id}/files/{path}       any output: clips, reel, heatmaps, preview

Run with ``padel-api`` (or ``uvicorn padel_api.main:create_app --factory``).
``padel-buddy`` (see ``local.py``) runs the API, a worker and the web app in
one process for use on your own computer.
"""

from __future__ import annotations

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
from .db import Database, Job, Match
from .schemas import (
    AnalyzeIn, CalibrationIn, CalibrationOut, JobOut, Keypoint, MatchCreate, MatchOut, UploadStatus,
)
from .settings import Settings, get_settings
from .storage import ALLOWED_VIDEO_EXTS, LocalStorage

_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_url)
    db.create_all()
    storage = LocalStorage(settings.data_dir)

    app = FastAPI(title="Padel Buddy API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    def get_session() -> Iterator[Session]:
        yield from db.session()

    def require_token(
        authorization: str | None = Header(None), token: str | None = Query(None)
    ) -> None:
        if not settings.api_token:
            return
        given = token or (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given, settings.api_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or wrong API token")

    def load_match(match_id: str, session: Session = Depends(get_session)) -> Match:
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

    # -- matches ----------------------------------------------------------

    @app.post("/matches", status_code=201, dependencies=auth)
    def create_match(body: MatchCreate, session: Session = Depends(get_session)) -> MatchOut:
        ext = Path(body.filename).suffix.lower() or ".mp4"
        if ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(415, f"Unsupported video type {ext}; use one of {sorted(ALLOWED_VIDEO_EXTS)}")
        _check_size(body.size, settings)
        match = Match(title=body.title, video_ext=ext, video_size=body.size)
        session.add(match)
        session.commit()
        storage.match_dir(match.id).mkdir(parents=True, exist_ok=True)
        return _match_out(match)

    @app.get("/matches", dependencies=auth)
    def list_matches(session: Session = Depends(get_session)) -> list[MatchOut]:
        rows = session.scalars(select(Match).order_by(Match.created_at.desc())).all()
        return [_match_out(m) for m in rows]

    @app.get("/matches/{match_id}", dependencies=auth)
    def get_match(match: Match = Depends(load_match)) -> MatchOut:
        return _match_out(match)

    @app.delete("/matches/{match_id}", status_code=204, dependencies=auth)
    def delete_match(match: Match = Depends(load_match), session: Session = Depends(get_session)) -> Response:
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
        session: Session = Depends(get_session),
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
        body: CalibrationIn, match: Match = Depends(load_match), session: Session = Depends(get_session)
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
    def clear_calibration(match: Match = Depends(load_match), session: Session = Depends(get_session)) -> Response:
        match = session.merge(match)
        match.calibration = None
        session.commit()
        return Response(status_code=204)

    # -- analysis and results ---------------------------------------------

    @app.post("/matches/{match_id}/analyze", status_code=202, dependencies=auth)
    def analyze(
        body: AnalyzeIn | None = None, match: Match = Depends(load_match), session: Session = Depends(get_session)
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
        for key, name in (("tracking", "summary.json"), ("clips", "clips.json")):
            f = run / name
            out[key] = json.loads(f.read_text()) if f.exists() else None
        return out

    @app.get("/matches/{match_id}/files/{rel:path}", dependencies=auth)
    def run_file(rel: str, match: Match = Depends(load_match)) -> FileResponse:
        path = storage.run_file(match.id, rel)
        if path is None:
            raise HTTPException(404, "No such output")
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


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id, status=job.status, progress=job.progress, error=job.error,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
    )


def _match_out(match: Match) -> MatchOut:
    latest = match.jobs[-1] if match.jobs else None
    return MatchOut(
        id=match.id, title=match.title, created_at=match.created_at, status=match.status,
        video_size=match.video_size, video_received=match.video_received, duration_s=match.duration_s,
        calibrated=match.calibration is not None, error=match.error,
        job=_job_out(latest) if latest else None,
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
