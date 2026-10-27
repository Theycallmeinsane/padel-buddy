"""A job queue in the database itself.

No Redis or broker to run: the API inserts a ``queued`` row and any number
of workers claim rows with a conditional UPDATE, which only one of them can
win. That works on SQLite for a single machine and on Postgres for many
workers, and the queue survives restarts because it is just a table.
"""

from __future__ import annotations

import time
from datetime import timedelta

from sqlalchemy import exists, select, update
from sqlalchemy.orm import Session, aliased

from .db import Job, Match, now

#   queued | running -> stopping -> stopped, when someone stops the analysis
ACTIVE = ("queued", "running", "stopping")
BUSY = ("running", "stopping")  # a worker is still writing the match's run folder


class Stopped(Exception):
    """The job was stopped (or its match deleted) while it ran."""


def enqueue(session: Session, match: Match, config: dict | None = None, kind: str = "analyze") -> Job:
    """Queue work on a match: a full analysis, or ``relabel`` (who is who again, after a tag).

    A job queued behind one already running for the same match waits for it.
    """
    running = any(j.status in BUSY for j in match.jobs)
    job = Job(match_id=match.id, kind=kind, config=config)
    match.jobs.append(job)
    if not running:
        match.status = "queued"
    match.error = None
    session.commit()
    return job


def active_job(session: Session, match_id: str) -> Job | None:
    return session.scalars(
        select(Job).where(Job.match_id == match_id, Job.status.in_(ACTIVE)).limit(1)
    ).first()


def claim_next(session: Session, worker: str) -> Job | None:
    """Atomically take the oldest queued job, or return None.

    Jobs for a match another worker is busy with wait, so two never write
    one run folder at once.
    """
    busy = aliased(Job)
    for _ in range(5):
        job_id = session.scalars(
            select(Job.id)
            .where(
                Job.status == "queued",
                ~exists().where(busy.match_id == Job.match_id, busy.status.in_(BUSY)),
            )
            .order_by(Job.created_at)
            .limit(1)
        ).first()
        if job_id is None:
            return None
        claimed = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .values(status="running", worker=worker, started_at=now(), heartbeat_at=now(), progress=0.0)
        )
        session.commit()
        if claimed.rowcount == 1:
            return session.get(Job, job_id)
        # Another worker won the race; try the next one.
    return None


def request_stop(session: Session, match: Match, stale_after_s: float) -> bool:
    """Stop the match's work: queued jobs never start, a running one is asked to stop.

    The worker sees the request the next time it reports progress, within a few
    seconds. A running job whose worker has gone quiet (the app was closed) is
    stopped straight away, as nobody is left to see the request. Returns
    whether a worker is still stopping.
    """
    cutoff = now() - timedelta(seconds=stale_after_s)
    quiet = set(session.scalars(
        select(Job.id).where(Job.match_id == match.id, Job.status.in_(BUSY), Job.heartbeat_at < cutoff)
    ))
    stopping, kinds = False, set()
    for job in match.jobs:
        if job.status == "queued" or job.id in quiet:
            job.status, job.finished_at = "stopped", now()
            kinds.add(job.kind)
        elif job.status in BUSY:
            job.status, stopping = "stopping", True
    if not stopping:
        match.status = _status_after_stop(kinds)
    session.commit()
    return stopping


def _status_after_stop(kinds: set[str]) -> str:
    # Who is who is decided again only on a finished analysis, whose results
    # are left as they were when that stops.
    return "done" if kinds == {"relabel"} else "stopped"


class ProgressReporter:
    """Writes progress at most every couple of seconds or percent.

    Each write also checks whether the job was asked to stop, and raises
    ``Stopped`` into the engine if so.
    """

    def __init__(self, session: Session, job: Job, min_interval_s: float = 2.0, min_step: float = 0.02):
        self.session, self.job = session, job
        self.min_interval_s, self.min_step = min_interval_s, min_step
        self._last_t = 0.0
        self._last_p = -1.0

    def __call__(self, p: float) -> None:
        t = time.monotonic()
        if p < 1.0 and p - self._last_p < self.min_step and t - self._last_t < self.min_interval_s:
            return
        # Read straight from the database: the API changes the status, not this session.
        if self.session.scalar(select(Job.status).where(Job.id == self.job.id)) != "running":
            raise Stopped
        self.job.progress = round(min(max(p, 0.0), 1.0), 4)
        self.job.heartbeat_at = now()
        self.session.commit()
        self._last_t, self._last_p = t, p


def finish(session: Session, job: Job, error: str | None = None, stopped: bool = False) -> None:
    job.status = "stopped" if stopped else "failed" if error else "done"
    job.error = error
    job.finished_at = now()
    if not error:
        job.progress = 1.0
    match = session.get(Match, job.match_id)
    if match is not None:
        waiting = any(j.status == "queued" for j in match.jobs if j.id != job.id)
        match.status = "queued" if waiting else _status_after_stop({job.kind}) if stopped else job.status
        match.error = error
    session.commit()


def requeue_stale(session: Session, stale_after_s: float) -> int:
    """Put back jobs whose worker died mid-run (no heartbeat for a while).

    One that was being stopped is stopped instead.
    """
    cutoff = now() - timedelta(seconds=stale_after_s)
    for job in session.scalars(select(Job).where(Job.status == "stopping", Job.heartbeat_at < cutoff)):
        finish(session, job, stopped=True)
    result = session.execute(
        update(Job)
        .where(Job.status == "running", Job.heartbeat_at < cutoff)
        .values(status="queued", worker=None, progress=0.0)
    )
    session.commit()
    return result.rowcount or 0
