"""Background worker: claims analysis jobs and runs the engine on them.

Run one per GPU (or one on your laptop):

    padel-worker
"""

from __future__ import annotations

import logging
import signal
import socket
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from . import jobs
from .db import Database, Job, Match
from .settings import Settings, get_settings
from .storage import LocalStorage

log = logging.getLogger("padel.worker")

# (video, run_dir, calibration | None, engine config, on_progress) -> anything
Runner = Callable[..., object]


def engine_runner(video: Path, run_dir: Path, calibration, config, on_progress) -> object:
    from padel_engine.pipeline import run_match

    return run_match(video, run_dir, calibration, config, on_progress=on_progress)


def process(session: Session, job: Job, storage: LocalStorage, runner: Runner = engine_runner) -> None:
    from padel_engine.calibration import Calibration
    from padel_engine.config import EngineConfig

    match = session.get(Match, job.match_id)
    if match is None:
        jobs.finish(session, job, error="match was deleted")
        return
    match.status = "processing"
    session.commit()
    try:
        config = EngineConfig.model_validate(job.config or {})
        calib = None
        if match.calibration:
            c = match.calibration
            calib = Calibration.from_points(
                {n: (float(u), float(v)) for n, (u, v) in c["points"].items()}, tuple(c["image_size"])
            )
        video = storage.video_path(match.id, match.video_ext)
        run_dir = storage.reset_run(match.id)
        runner(video, run_dir, calib, config, jobs.ProgressReporter(session, job))
    except Exception as exc:  # the job records the failure; the worker keeps going
        log.error("job %s failed:\n%s", job.id, traceback.format_exc())
        session.rollback()
        jobs.finish(session, job, error=f"{type(exc).__name__}: {exc}")
        return
    jobs.finish(session, job)


def run_once(db: Database, storage: LocalStorage, worker: str, runner: Runner = engine_runner) -> bool:
    """Process one job if there is one. Returns whether a job ran."""
    with db.sessions() as session:
        job = jobs.claim_next(session, worker)
        if job is None:
            return False
        log.info("running job %s for match %s", job.id, job.match_id)
        process(session, job, storage, runner)
        return True


def run_forever(settings: Settings, stop: threading.Event) -> None:
    """Poll for jobs until ``stop`` is set (checked between jobs)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)  # the worker may start before the API
    db = Database(settings.db_url)
    db.create_all()
    storage = LocalStorage(settings.data_dir)
    name = f"{socket.gethostname()}:{id(db)}"

    log.info("worker %s polling %s", name, settings.db_url.split("@")[-1])
    while not stop.is_set():
        with db.sessions() as s:
            if n := jobs.requeue_stale(s, settings.stale_job_s):
                log.warning("re-queued %d stale job(s)", n)
        if not run_once(db, storage, name):
            stop.wait(settings.worker_poll_s)


def main(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stop = threading.Event()

    def on_signal(*_):
        stop.set()
        log.info("stopping after the current job")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    run_forever(settings or get_settings(), stop)


if __name__ == "__main__":  # pragma: no cover
    main()
