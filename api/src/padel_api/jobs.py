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

ACTIVE = ("queued", "running")


def enqueue(session: Session, match: Match, config: dict | None = None, kind: str = "analyze") -> Job:
    """Queue work on a match: a full analysis, or ``relabel`` (who is who again, after a tag).

    A job queued behind one already running for the same match waits for it.
    """
    running = any(j.status == "running" for j in match.jobs)
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
                ~exists().where(busy.match_id == Job.match_id, busy.status == "running"),
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


class ProgressReporter:
    """Writes progress at most every couple of seconds or percent."""

    def __init__(self, session: Session, job: Job, min_interval_s: float = 2.0, min_step: float = 0.02):
        self.session, self.job = session, job
        self.min_interval_s, self.min_step = min_interval_s, min_step
        self._last_t = 0.0
        self._last_p = -1.0

    def __call__(self, p: float) -> None:
        t = time.monotonic()
        if p < 1.0 and p - self._last_p < self.min_step and t - self._last_t < self.min_interval_s:
            return
        self.job.progress = round(min(max(p, 0.0), 1.0), 4)
        self.job.heartbeat_at = now()
        self.session.commit()
        self._last_t, self._last_p = t, p


def finish(session: Session, job: Job, error: str | None = None) -> None:
    job.status = "failed" if error else "done"
    job.error = error
    job.finished_at = now()
    if not error:
        job.progress = 1.0
    match = session.get(Match, job.match_id)
    if match is not None:
        waiting = any(j.status == "queued" for j in match.jobs if j.id != job.id)
        match.status = "queued" if waiting else job.status
        match.error = error
    session.commit()


def requeue_stale(session: Session, stale_after_s: float) -> int:
    """Put back jobs whose worker died mid-run (no heartbeat for a while)."""
    cutoff = now() - timedelta(seconds=stale_after_s)
    result = session.execute(
        update(Job)
        .where(Job.status == "running", Job.heartbeat_at < cutoff)
        .values(status="queued", worker=None, progress=0.0)
    )
    session.commit()
    return result.rowcount or 0
