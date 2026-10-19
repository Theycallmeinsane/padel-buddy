import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from padel_api import jobs, worker
from padel_api.db import Database
from padel_api.main import create_app
from padel_api.settings import Settings
from padel_api.storage import LocalStorage

ENGINE_TESTS = Path(__file__).resolve().parents[2] / "engine" / "tests"
sys.path.insert(0, str(ENGINE_TESTS))

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", api_token=None, worker_poll_s=0.01)


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


def make_video(path: Path, seconds: float = 3.0) -> Path:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(int(seconds * 10)):
        w.write(np.full((48, 64, 3), i * 8 % 255, np.uint8))
    w.release()
    return path


def upload_in_chunks(client: TestClient, match_id: str, data: bytes, chunk: int) -> dict:
    total = len(data)
    out = None
    for start in range(0, total, chunk):
        end = min(start + chunk, total) - 1
        r = client.put(
            f"/matches/{match_id}/video",
            content=data[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{total}"},
        )
        assert r.status_code == 200, r.text
        out = r.json()
    return out


def test_create_upload_resume_and_frame(client, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "Sunday doubles", "filename": "GoPro.MP4"}).json()
    assert m["status"] == "created"

    # First chunk, then a "dropped connection": a chunk past the stored bytes is refused.
    half = len(data) // 2
    r = client.put(f"/matches/{m['id']}/video", content=data[:half],
                   headers={"Content-Range": f"bytes 0-{half - 1}/{len(data)}"})
    assert r.json() == {"received": half, "total": len(data), "complete": False}
    r = client.put(f"/matches/{m['id']}/video", content=data[half + 10 :],
                   headers={"Content-Range": f"bytes {half + 10}-{len(data) - 1}/{len(data)}"})
    assert r.status_code == 409

    # Resume from where the server says.
    received = client.get(f"/matches/{m['id']}/video/status").json()["received"]
    r = client.put(f"/matches/{m['id']}/video", content=data[received:],
                   headers={"Content-Range": f"bytes {received}-{len(data) - 1}/{len(data)}"})
    assert r.json()["complete"] is True

    match = client.get(f"/matches/{m['id']}").json()
    assert match["status"] == "uploaded"
    assert match["duration_s"] == pytest.approx(3.0, abs=0.2)
    assert client.get(f"/matches/{m['id']}/video").content == data

    f = client.get(f"/matches/{m['id']}/frame", params={"t": 1})
    assert f.status_code == 200 and f.headers["content-type"] == "image/jpeg"
    assert f.headers["x-frame-width"] == "64"


def test_single_shot_upload_and_rejections(client, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    assert client.post("/matches", json={"title": "x", "filename": "notes.txt"}).status_code == 415
    m = client.post("/matches", json={"title": "x"}).json()
    assert client.post(f"/matches/{m['id']}/analyze").status_code == 409  # nothing uploaded
    r = client.put(f"/matches/{m['id']}/video", content=data)
    assert r.json()["complete"] is True
    assert client.put(f"/matches/{m['id']}/video", content=data).status_code == 409  # already done
    bad = client.post(f"/matches/{m['id']}/analyze", json={"config": {"clips": {"pre_s": -1}}})
    assert bad.status_code == 422
    assert client.get("/matches/nope").status_code == 404


def test_calibration_validation(client):
    from padel_engine import court

    m = client.post("/matches", json={"title": "x"}).json()
    # A plausible behind-the-court view: far wall narrow and high, service lines wider and lower.
    pts = {
        "far_left_corner": (400, 200), "far_right_corner": (880, 200),
        "far_service_left": (370, 240), "far_service_right": (910, 240),
        "near_service_left": (60, 640), "near_service_right": (1220, 640),
    }
    assert set(pts) <= set(court.KEYPOINTS)
    r = client.put(f"/matches/{m['id']}/calibration", json={"image_size": [1280, 720], "points": pts})
    assert r.status_code == 200, r.text
    assert client.get(f"/matches/{m['id']}").json()["calibrated"] is True

    bad = {"far_left_corner": (1, 1), "far_right_corner": (2, 2), "far_service_left": (3, 3)}
    r = client.put(f"/matches/{m['id']}/calibration", json={"image_size": [1280, 720], "points": bad})
    assert r.status_code == 422
    assert len(client.get("/court/keypoints").json()) == len(court.KEYPOINTS)


def test_job_lifecycle_with_fake_engine(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "x"}).json()
    client.put(f"/matches/{m['id']}/video", content=data)

    r = client.post(f"/matches/{m['id']}/analyze", json={"config": {"clips": {"pre_s": 20}}})
    assert r.status_code == 202
    assert client.post(f"/matches/{m['id']}/analyze").status_code == 409  # already queued

    seen = {}

    def fake_runner(video, run_dir, calibration, config, on_progress):
        seen.update(video=video, calibration=calibration, pre=config.clips.pre_s)
        on_progress(0.5)
        (run_dir / "clips.json").write_text(json.dumps({"clips": [{"file": "clips/clip_001.mp4"}]}))
        (run_dir / "clips").mkdir()
        (run_dir / "clips" / "clip_001.mp4").write_bytes(b"fake")

    db = Database(settings.db_url)
    storage = LocalStorage(settings.data_dir)
    assert worker.run_once(db, storage, "test", fake_runner) is True
    assert worker.run_once(db, storage, "test", fake_runner) is False  # queue empty

    assert seen["pre"] == 20 and seen["calibration"] is None and Path(seen["video"]).exists()
    match = client.get(f"/matches/{m['id']}").json()
    assert match["status"] == "done" and match["job"]["progress"] == 1.0
    res = client.get(f"/matches/{m['id']}/results").json()
    assert res["clips"]["clips"][0]["file"] == "clips/clip_001.mp4" and res["tracking"] is None
    assert client.get(f"/matches/{m['id']}/files/clips/clip_001.mp4").content == b"fake"
    assert client.get(f"/matches/{m['id']}/files/../video.mp4").status_code == 404


def test_failed_job_is_recorded_and_can_rerun(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "x"}).json()
    client.put(f"/matches/{m['id']}/video", content=data)
    client.post(f"/matches/{m['id']}/analyze")

    def broken(*_args, **_kw):
        raise RuntimeError("GPU on fire")

    db = Database(settings.db_url)
    worker.run_once(db, LocalStorage(settings.data_dir), "test", broken)
    match = client.get(f"/matches/{m['id']}").json()
    assert match["status"] == "failed" and "GPU on fire" in match["job"]["error"]
    assert client.post(f"/matches/{m['id']}/analyze").status_code == 202


def test_only_one_worker_claims_a_job(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "x"}).json()
    client.put(f"/matches/{m['id']}/video", content=data)
    client.post(f"/matches/{m['id']}/analyze")
    db = Database(settings.db_url)
    with db.sessions() as a, db.sessions() as b:
        first = jobs.claim_next(a, "w1")
        second = jobs.claim_next(b, "w2")
    assert first is not None and second is None


def test_stale_running_job_is_requeued(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "x"}).json()
    client.put(f"/matches/{m['id']}/video", content=data)
    client.post(f"/matches/{m['id']}/analyze")
    db = Database(settings.db_url)
    with db.sessions() as s:
        jobs.claim_next(s, "w1")
        assert jobs.requeue_stale(s, stale_after_s=3600) == 0
        assert jobs.requeue_stale(s, stale_after_s=-1) == 1
        assert jobs.claim_next(s, "w2") is not None


def test_token_required_when_configured(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path / "d", api_token="s3cret")))
    assert c.get("/health").status_code == 200
    assert c.get("/matches").status_code == 401
    assert c.get("/matches", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/matches", params={"token": "s3cret"}).status_code == 200


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_real_engine_end_to_end(client, settings, tmp_path):
    """Upload a synthetic match, run the real engine (clips only), fetch a clip."""
    from synth_audio import default_script
    from synth_audio import make_video as make_match

    data = make_match(tmp_path / "match.mp4", default_script()).read_bytes()
    m = client.post("/matches", json={"title": "synthetic"}).json()
    upload_in_chunks(client, m["id"], data, chunk=256 * 1024)
    assert client.post(f"/matches/{m['id']}/analyze").status_code == 202

    assert worker.run_once(Database(settings.db_url), LocalStorage(settings.data_dir), "test")
    match = client.get(f"/matches/{m['id']}").json()
    assert match["status"] == "done", match
    clips = client.get(f"/matches/{m['id']}/results").json()["clips"]
    assert len(clips["rallies"]) == 3 and len(clips["clips"]) == 3 and clips["condensed"] == "condensed.mp4"
    clip = client.get(f"/matches/{m['id']}/files/{clips['clips'][0]['file']}")
    assert clip.status_code == 200 and len(clip.content) > 10_000
    ranged = client.get(f"/matches/{m['id']}/files/highlights.mp4", headers={"Range": "bytes=0-99"})
    assert ranged.status_code == 206 and len(ranged.content) == 100
    data = client.get(f"/matches/{m['id']}/files/clips.json")
    assert data.headers["content-encoding"] == "gzip" and data.json()["condensed"] == "condensed.mp4"
    assert len(data.json()["condensed_segments"]) == 3


def test_worker_starts_before_the_api_has_created_anything(tmp_path, monkeypatch):
    """The worker must create the data folder itself (it used to crash on a fresh machine)."""
    settings = Settings(data_dir=tmp_path / "fresh" / "data", worker_poll_s=0.01)
    calls = []

    def fake_run_once(db, storage, name, runner=None):
        calls.append(name)
        raise SystemExit  # stop the loop after one pass

    monkeypatch.setattr(worker, "run_once", fake_run_once)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    with pytest.raises(SystemExit):
        worker.main(settings)
    assert calls and (tmp_path / "fresh" / "data" / "padel.db").exists()


def test_serves_the_web_app_next_to_the_api(tmp_path):
    web = tmp_path / "out"
    (web / "match").mkdir(parents=True)
    (web / "index.html").write_text("<h1>Matches</h1>")
    (web / "match" / "index.html").write_text("<h1>Match</h1>")
    c = TestClient(create_app(Settings(data_dir=tmp_path / "d", web_dir=web)))
    assert "Matches" in c.get("/").text
    assert "Match" in c.get("/match/", params={"id": "abc"}).text
    assert c.get("/health").json()["ok"] is True  # API routes still win
    assert c.get("/matches").json() == []


def test_web_pages_and_api_routes_never_share_a_path(settings):
    # API routes win when one process serves both, so a page at /people/ would lose
    # its own files (people/__next._tree.txt) to GET /people/{name}.
    app_dir = Path(__file__).resolve().parents[2] / "web" / "app"
    pages = {p.name for p in app_dir.iterdir() if (p / "page.tsx").exists()}
    api = {route.path.strip("/").split("/")[0] for route in create_app(settings).routes}
    assert pages and not pages & api


def test_import_a_video_already_on_disk(settings, tmp_path):
    from padel_api.main import import_video

    src = make_video(tmp_path / "IMG_0257.MOV")
    match = import_video(settings, src)
    assert match.status == "uploaded" and match.title == "IMG_0257"
    assert match.duration_s == pytest.approx(3.0, abs=0.2)
    stored = LocalStorage(settings.data_dir).video_path(match.id, ".mov")
    assert stored.read_bytes() == src.read_bytes()
    c = TestClient(create_app(settings))
    assert [m["id"] for m in c.get("/matches").json()] == [match.id]
    c.delete(f"/matches/{match.id}")
    assert src.exists()  # deleting the match never touches the original
    with pytest.raises(ValueError):
        import_video(settings, tmp_path / "notes.txt")


def test_worker_loop_stops_when_asked(settings, tmp_path, monkeypatch):
    import threading

    stop = threading.Event()
    seen = []

    def fake_run_once(db, storage, name, runner=None):
        seen.append(name)
        stop.set()
        return False

    monkeypatch.setattr(worker, "run_once", fake_run_once)
    t = threading.Thread(target=worker.run_forever, args=(settings, stop))
    t.start()
    t.join(timeout=5)
    assert not t.is_alive() and len(seen) == 1


def test_padel_buddy_import_command_queues_analysis(settings, tmp_path, capsys):
    from padel_api import local

    local.main(["import", str(make_video(tmp_path / "match.mp4")), "--title", "Sunday", "--analyze"], settings)
    assert "Added 'Sunday'" in capsys.readouterr().out
    (m,) = TestClient(create_app(settings)).get("/matches").json()
    assert m["status"] == "queued" and m["job"]["status"] == "queued"


def _fake_report_runner(video, run_dir, calibration, config, on_progress):
    """Writes the shots and report the engine would make for the scripted match in engine/tests."""
    import match_fixture as mf
    from padel_engine.report import build_report
    from padel_engine.schemas import ShotsSummary
    from padel_engine.shots import Ball, NetBand, Players, Rally, find_shots

    overlay = mf.overlay()
    shots, points = find_shots(mf.hits(), [Rally(*r) for r in mf.RALLIES], Players(overlay),
                               Ball(mf.ball_track()), NetBand(overlay["court_to_image"]))
    summary = ShotsSummary(engine_version="test", shots=shots, points=points, outputs={})
    (run_dir / "shots.json").write_text(summary.model_dump_json())
    report = build_report(summary, overlay, mf.clips())
    report.players[2].reel = "players/P3.mp4"
    (run_dir / "report.json").write_text(report.model_dump_json())


def test_players_are_named_and_found_across_matches(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "Sunday"}).json()
    client.put(f"/matches/{m['id']}/video", content=data)
    client.post(f"/matches/{m['id']}/analyze")
    assert worker.run_once(Database(settings.db_url), LocalStorage(settings.data_dir), "test", _fake_report_runner)

    r = client.put(f"/matches/{m['id']}/players", json={"names": {"P3": "Maya", "P1": "  Ali   Khan "}})
    assert r.status_code == 200 and r.json()["player_names"] == {"P1": "Ali Khan", "P3": "Maya"}
    assert client.put(f"/matches/{m['id']}/players", json={"names": {"P5": "x"}}).status_code == 422

    res = client.get(f"/matches/{m['id']}/results").json()
    assert res["report"]["points"] == 3 and len(res["shots"]["shots"]) == 14

    p3 = client.get(f"/matches/{m['id']}/players/P3").json()
    assert (p3["name"], p3["team"], p3["reel"]) == ("Maya", "B", "players/P3.mp4")
    assert p3["report"]["winners"] == 1 and {s["player"] for s in p3["shots"]} == {"P3"}
    assert client.get(f"/matches/{m['id']}/players/P9").status_code == 404

    assert client.get("/people").json() == [{"name": "Ali Khan", "matches": 1}, {"name": "Maya", "matches": 1}]
    maya = client.get("/people/maya").json()
    assert [(h["match_title"], h["player"]) for h in maya] == [("Sunday", "P3")]
    assert client.get("/people/nobody").status_code == 404

    md = client.get(f"/matches/{m['id']}/report.md")
    assert md.status_code == 200 and "Maya (P3)" in md.text

    client.put(f"/matches/{m['id']}/players", json={"names": {"P1": ""}})
    assert client.get(f"/matches/{m['id']}").json()["player_names"] == {"P3": "Maya"}


def test_columns_added_in_newer_versions_reach_an_old_database(settings):
    from sqlalchemy import inspect, text

    settings.data_dir.mkdir(parents=True)
    db = Database(settings.db_url)
    db.create_all()
    with db.engine.begin() as conn:
        conn.execute(text("ALTER TABLE matches DROP COLUMN player_names"))
    Database(settings.db_url).create_all()
    assert "player_names" in {c["name"] for c in inspect(db.engine).get_columns("matches")}


def test_the_claude_key_in_env_reaches_the_engine(tmp_path, monkeypatch):
    import os
    import threading

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-test\n")
    settings = Settings(_env_file=env, data_dir=tmp_path / "data")
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"
    stop = threading.Event()
    stop.set()
    worker.run_forever(settings, stop)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test"
