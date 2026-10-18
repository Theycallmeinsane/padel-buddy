"""Database models: matches and analysis jobs.

The same models run on SQLite (local) and Postgres (Supabase or any host);
only the URL changes.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(default=now)
    # created -> uploading -> uploaded -> queued -> processing -> done | failed
    status: Mapped[str] = mapped_column(String(20), default="created")
    video_ext: Mapped[str] = mapped_column(String(8), default=".mp4")
    video_size: Mapped[int | None] = mapped_column(Integer, nullable=True)  # total bytes expected
    video_received: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # engine Calibration.to_dict()
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    jobs: Mapped[list[Job]] = relationship(
        back_populates="match", cascade="all, delete-orphan", order_by="Job.created_at"
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="analyze")
    # queued -> running -> done | failed
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # EngineConfig overrides
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=now)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    match: Mapped[Match] = relationship(back_populates="jobs")


class Database:
    def __init__(self, url: str):
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        try:
            Base.metadata.create_all(self.engine)
        except OperationalError:
            # The API and a worker starting together on a fresh database can
            # both try to create a table; the second pass finds them all.
            Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        with self.sessions() as s:
            yield s


def _sqlite_pragmas(conn, _record) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")  # API and worker read/write concurrently
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()
