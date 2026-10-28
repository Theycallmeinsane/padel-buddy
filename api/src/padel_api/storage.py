"""Where a match's video and run folder live.

Locally that is ``data_dir/matches/<id>/``. A cloud deployment keeps the
same layout on a disk the API and workers share, or swaps this class for
one that syncs the folder with object storage (Supabase Storage, S3); the
rest of the API only talks to this interface.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


class LocalStorage:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir) / "matches"
        self.root.mkdir(parents=True, exist_ok=True)

    def match_dir(self, match_id: str) -> Path:
        return self.root / match_id

    def video_path(self, match_id: str, ext: str) -> Path:
        return self.match_dir(match_id) / f"video{ext}"

    def web_video_path(self, match_id: str) -> Path:
        """An H.264 copy of the video, made when browsers cannot show the original."""
        return self.match_dir(match_id) / "video.web.mp4"

    def run_dir(self, match_id: str) -> Path:
        return self.match_dir(match_id) / "run"

    def run_file(self, match_id: str, rel: str) -> Path | None:
        """Resolve a path inside the run folder, refusing anything outside it."""
        base = self.run_dir(match_id).resolve()
        target = (base / rel).resolve()
        if base not in target.parents or not target.is_file():
            return None
        return target

    def reset_run(self, match_id: str) -> Path:
        run = self.run_dir(match_id)
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        return run

    def delete(self, match_id: str) -> None:
        shutil.rmtree(self.match_dir(match_id), ignore_errors=True)
