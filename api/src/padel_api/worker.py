"""Background worker: claims analysis jobs and runs the engine on them.

Run one per GPU (or one on your laptop):

    padel-worker
"""

from __future__ import annotations

import json
import logging
import os
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

# (video, run_dir, calibration | None, engine config, on_progress, tags=[...]) -> anything
Runner = Callable[..., object]


def engine_runner(video: Path, run_dir: Path, calibration, config, on_progress, tags=()) -> object:
    from padel_engine.pipeline import run_match
    from padel_engine.tags import Tag

    return run_match(
        video, run_dir, calibration, config, on_progress=on_progress, tags=[Tag.from_dict(t) for t in tags]
    )


def engine_relabeler(video: Path, run_dir: Path, calibration, config, on_progress, tags=()) -> object:
    from padel_engine.pipeline import apply_tags
    from padel_engine.tags import Tag

    return apply_tags(video, run_dir, calibration, [Tag.from_dict(t) for t in tags], config, on_progress)


def process(
    session: Session, job: Job, storage: LocalStorage, runner: Runner = engine_runner,
    relabeler: Runner = engine_relabeler,
) -> None:
    from padel_engine.calibration import Calibration
    from padel_engine.config import EngineConfig

    job_id, match_id = job.id, job.match_id  # the rows may be gone after a stop
    match = session.get(Match, match_id)
    if match is None:
        jobs.finish(session, job, error="match was deleted")
        return
    match.status = "processing"
    session.commit()
    tags = list(match.tags or [])
    try:
        config = EngineConfig.model_validate(job.config or {})
        calib = None
        if match.calibration:
            c = match.calibration
            calib = Calibration.from_points(
                {n: (float(u), float(v)) for n, (u, v) in c["points"].items()}, tuple(c["image_size"])
            )
        video = storage.video_path(match.id, match.video_ext)
        if job.kind == "relabel":
            if calib is None:
                raise ValueError("Tags need the court calibrated and the match analysed")
            relabeler(video, storage.run_dir(match.id), calib, config, jobs.ProgressReporter(session, job), tags=tags)
        else:
            run_dir = storage.reset_run(match.id)
            runner(video, run_dir, calib, config, jobs.ProgressReporter(session, job), tags=tags)
        _take_names(session, match, tags, storage.run_dir(match.id) / "summary.json")
    except jobs.Stopped:
        session.rollback()
        log.info("job %s stopped", job_id)
        job = session.get(Job, job_id)
        if job is not None:
            jobs.finish(session, job, stopped=True)
        else:
            storage.delete(match_id)  # the match was deleted while it ran: what the job wrote goes too
        return
    except Exception as exc:  # the job records the failure; the worker keeps going
        log.error("job %s failed:\n%s", job.id, traceback.format_exc())
        session.rollback()
        jobs.finish(session, job, error=f"{type(exc).__name__}: {exc}")
        return
    jobs.finish(session, job)


def _take_names(session: Session, match: Match, tags: list[dict], summary_path: Path) -> None:
    """Give players the names their tags settled on, and mark which tags held.

    ``tags`` is the list the engine was given; tags added meanwhile wait for
    the job queued after them. A name a tag puts on one player is taken off
    any other; names given without a tag stay where they are otherwise.
    """
    if not tags or not summary_path.exists():
        return
    session.refresh(match)  # tags and names may have changed while the job ran
    summary = json.loads(summary_path.read_bytes())
    named = summary.get("names", {})
    missed = set(summary.get("unmatched_tags", []))
    held = {t["id"]: k not in missed for k, t in enumerate(tags)}
    match.tags = [{**t, "applied": held.get(t["id"], t.get("applied"))} for t in match.tags or []]
    names = {
        pid: n for pid, n in (match.player_names or {}).items()
        if n.casefold() not in {v.casefold() for v in named.values()}
    }
    names.update(named)
    match.player_names = names or None


def run_once(
    db: Database, storage: LocalStorage, worker: str, runner: Runner = engine_runner,
    relabeler: Runner = engine_relabeler,
) -> bool:
    """Process one job if there is one. Returns whether a job ran."""
    with db.sessions() as session:
        job = jobs.claim_next(session, worker)
        if job is None:
            return False
        log.info("running %s job %s for match %s", job.kind, job.id, job.match_id)
        process(session, job, storage, runner, relabeler)
        return True


def run_forever(settings: Settings, stop: threading.Event) -> None:
    """Poll for jobs until ``stop`` is set (checked between jobs)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)  # the worker may start before the API
    if settings.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        # The engine reads the key from the environment; .env alone doesn't put it there.
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key.get_secret_value()
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
