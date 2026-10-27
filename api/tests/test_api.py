import json
import shutil
import sys
from datetime import timedelta
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

    def fake_runner(video, run_dir, calibration, config, on_progress, tags=()):
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


def _fake_report_runner(video, run_dir, calibration, config, on_progress, tags=()):
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
    from padel_engine.schemas import Short

    report.players[2].reel = "players/P3.mp4"
    report.players[2].winners_reel = "players/P3_winners.mp4"
    report.shorts = [
        Short(file="shorts/point_001.mp4", rally=0, start_s=0, end_s=12, title="Point 1"),
        Short(file="shorts/P3_winner.mp4", rally=0, start_s=2, end_s=13, title="P3's winner", player="P3"),
    ]
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
    assert p3["winners_reel"] == "players/P3_winners.mp4"
    assert [x["file"] for x in p3["shorts"]] == ["shorts/P3_winner.mp4"]
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


COURT = {
    "far_left_corner": (400, 200), "far_right_corner": (880, 200),
    "far_service_left": (370, 240), "far_service_right": (910, 240),
    "near_service_left": (60, 640), "near_service_right": (1220, 640),
}


def test_tags_name_players_and_fix_mix_ups(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "Sunday"}).json()
    mid = m["id"]
    client.put(f"/matches/{mid}/video", content=data)
    client.put(f"/matches/{mid}/calibration", json={"image_size": [1280, 720], "points": COURT})

    # Named on the setup frame, before any analysis: the tag waits for it.
    r = client.post(f"/matches/{mid}/tags", json={"time_s": 5, "x": 300, "y": 500, "name": " Maya "})
    assert r.status_code == 201, r.text
    (tag,) = r.json()["tags"]
    assert tag["name"] == "Maya" and tag["applied"] is None and r.json()["player_names"] == {}

    given = []

    def engine(names, unmatched=()):
        """A stand-in engine: records the tags and names players as told."""
        def run(video, run_dir, calibration, config, on_progress, tags=()):
            given.append([t["name"] for t in tags])
            (run_dir / "segments.json").write_text("{}")
            (run_dir / "summary.json").write_text(json.dumps({"names": names, "unmatched_tags": list(unmatched)}))
        return run

    db, storage = Database(settings.db_url), LocalStorage(settings.data_dir)
    client.post(f"/matches/{mid}/analyze")
    assert worker.run_once(db, storage, "test", engine({"P1": "Maya"}))
    m = client.get(f"/matches/{mid}").json()
    assert given == [["Maya"]] and m["player_names"] == {"P1": "Maya"} and m["tags"][0]["applied"] is True

    # Watching: tagging a new name on a box names that player at once, no job needed.
    r = client.post(f"/matches/{mid}/tags", json={"time_s": 60, "x": 900, "y": 300, "name": "Ali", "player": "P3"})
    m = r.json()
    assert m["player_names"] == {"P1": "Maya", "P3": "Ali"} and m["job"]["kind"] == "analyze"

    # Maya was tagged as P1; a tag says the P2 box is Maya: tracking mixed them up.
    r = client.post(f"/matches/{mid}/tags", json={"time_s": 600, "x": 700, "y": 600, "name": "maya", "player": "P2"})
    m = r.json()
    assert m["player_names"] == {"P1": "Maya", "P3": "Ali"}  # unchanged until who is who is decided again
    assert m["job"]["kind"] == "relabel" and m["job"]["status"] == "queued" and m["status"] == "queued"
    assert m["tags"][-1]["applied"] is None

    relabelled = []

    def relabeler(video, run_dir, calibration, config, on_progress, tags=()):
        relabelled.append((calibration is not None, [t["name"] for t in tags]))
        (run_dir / "summary.json").write_text(json.dumps({"names": {"P1": "Maya", "P3": "Ali"}, "unmatched_tags": [2]}))

    assert worker.run_once(db, storage, "test", engine({}), relabeler)
    m = client.get(f"/matches/{mid}").json()
    assert relabelled == [(True, ["Maya", "Ali", "Maya"])] and m["status"] == "done"  # one spelling per person
    assert [t["applied"] for t in m["tags"]] == [True, True, False]  # the engine could not apply the last one

    # Four players: a fifth name is refused.
    for name, pid in (("Sam", "P2"), ("Jo", "P4")):
        assert client.post(f"/matches/{mid}/tags", json={"time_s": 1, "x": 1, "y": 1, "name": name, "player": pid}).status_code == 201
    r = client.post(f"/matches/{mid}/tags", json={"time_s": 1, "x": 1, "y": 1, "name": "Kim", "player": "P2"})
    assert r.status_code == 409

    # Renaming a player renames their tags; taking back a tag that held decides who is who again.
    m = client.put(f"/matches/{mid}/players", json={"names": {"P3": "Ali Khan"}}).json()
    assert [t["name"] for t in m["tags"]].count("Ali Khan") == 1
    assert client.delete(f"/matches/{mid}/tags/nope").status_code == 404
    m = client.delete(f"/matches/{mid}/tags/{m['tags'][0]['id']}").json()
    assert len(m["tags"]) == 4 and m["job"]["kind"] == "relabel"


def test_a_match_s_jobs_run_one_at_a_time(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "x"}).json()
    client.put(f"/matches/{m['id']}/video", content=data)
    client.post(f"/matches/{m['id']}/analyze")
    db = Database(settings.db_url)
    with db.sessions() as a, db.sessions() as b:
        first = jobs.claim_next(a, "one")
        first.match.status = "processing"  # as the worker does
        a.commit()
        match = b.get(type(first.match), m["id"])
        jobs.enqueue(b, match, kind="relabel")
        assert match.status == "processing"  # still the running analysis
        assert jobs.claim_next(b, "two") is None  # waits for the analysis
        jobs.finish(a, first)
        assert a.get(type(first.match), m["id"]).status == "queued"  # the relabel is next
        assert jobs.claim_next(b, "two").kind == "relabel"


def test_results_so_far_are_served_while_analysing_and_after_a_failure(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    mid = client.post("/matches", json={"title": "x"}).json()["id"]
    client.put(f"/matches/{mid}/video", content=data)
    client.put(f"/matches/{mid}/calibration", json={"image_size": [1280, 720], "points": COURT})
    client.post(f"/matches/{mid}/analyze")
    seen = []

    def engine(video, run_dir, calibration, config, on_progress, tags=()):
        (run_dir / "clips.json").write_text(json.dumps({"rallies": [{"index": 0}, {"index": 1}], "clips": []}))
        (run_dir / "live.json").write_text(json.dumps({"covered_s": 300.0, "duration_s": 5400.0, "done": False}))
        on_progress(0.2)
        seen.append((client.get(f"/matches/{mid}").json(), client.get(f"/matches/{mid}/results").json()))
        raise RuntimeError("power cut")

    worker.run_once(Database(settings.db_url), LocalStorage(settings.data_dir), "test", engine)
    match, results = seen[0]
    assert match["status"] == "processing" and match["live"] == {"covered_s": 300.0, "duration_s": 5400.0}
    assert len(results["clips"]["rallies"]) == 2
    # The analysis failed, but what it published stays readable.
    match = client.get(f"/matches/{mid}").json()
    assert match["status"] == "failed" and match["live"]["covered_s"] == 300.0
    assert len(client.get(f"/matches/{mid}/results").json()["clips"]["rallies"]) == 2

    def finished(video, run_dir, calibration, config, on_progress, tags=()):
        (run_dir / "live.json").write_text(json.dumps({"covered_s": 5400.0, "duration_s": 5400.0, "done": True}))

    client.post(f"/matches/{mid}/analyze")
    worker.run_once(Database(settings.db_url), LocalStorage(settings.data_dir), "test", finished)
    match = client.get(f"/matches/{mid}").json()
    assert match["status"] == "done" and match["live"] is None


def test_streaming_a_video_or_clip_holds_no_database_connection(client, settings, tmp_path):
    # A browser keeps video and clip streams open for as long as they play. A
    # connection held for each stream ran the pool dry ("QueuePool limit reached").
    data = make_video(tmp_path / "v.mp4").read_bytes()
    m = client.post("/matches", json={"title": "Sunday doubles"}).json()
    upload_in_chunks(client, m["id"], data, len(data))
    clip = LocalStorage(settings.data_dir).run_dir(m["id"]) / "clips" / "rally_001.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(data)

    app, in_use = client.app, []

    async def watch(scope, receive, send):
        async def spy(message):
            if message["type"] == "http.response.body":
                in_use.append(app.state.db.engine.pool.checkedout())
            await send(message)

        await app(scope, receive, spy)

    watched = TestClient(watch)
    for url in (f"/matches/{m['id']}/video", f"/matches/{m['id']}/files/clips/rally_001.mp4"):
        r = watched.get(url)
        assert r.status_code == 200 and r.content == data
    assert in_use and max(in_use) == 0


def test_stopping_drops_queued_work_and_stops_a_running_job(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    mid = client.post("/matches", json={"title": "x"}).json()["id"]
    client.put(f"/matches/{mid}/video", content=data)
    assert client.post(f"/matches/{mid}/stop").status_code == 409  # nothing to stop

    # Queued: it never starts.
    client.post(f"/matches/{mid}/analyze")
    stopped = client.post(f"/matches/{mid}/stop").json()
    assert stopped["status"] == "stopped" and stopped["job"]["status"] == "stopped"
    db, storage = Database(settings.db_url), LocalStorage(settings.data_dir)
    assert worker.run_once(db, storage, "test", lambda *a, **k: None) is False

    # Running: it stops at its next progress report, and what it found so far stays.
    client.post(f"/matches/{mid}/analyze")
    reached = []

    def engine(video, run_dir, calibration, config, on_progress, tags=()):
        (run_dir / "live.json").write_text(json.dumps({"covered_s": 300.0, "duration_s": 5400.0, "done": False}))
        on_progress(0.1)
        match = client.post(f"/matches/{mid}/stop").json()
        assert match["status"] == "processing" and match["job"]["status"] == "stopping"
        assert client.post(f"/matches/{mid}/analyze").status_code == 409  # not until it has stopped
        on_progress(0.5)
        reached.append("past the stop")

    assert worker.run_once(db, storage, "test", engine) is True
    assert reached == []
    match = client.get(f"/matches/{mid}").json()
    assert match["status"] == "stopped" and match["job"]["status"] == "stopped" and match["error"] is None
    assert match["live"]["covered_s"] == 300.0
    assert client.post(f"/matches/{mid}/analyze").status_code == 202  # and it can run again


def test_deleting_a_match_being_analysed_stops_it_first(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    mid = client.post("/matches", json={"title": "x"}).json()["id"]
    client.put(f"/matches/{mid}/video", content=data)
    client.post(f"/matches/{mid}/analyze")
    answers = []

    def engine(video, run_dir, calibration, config, on_progress, tags=()):
        on_progress(0.1)
        answers.append(client.delete(f"/matches/{mid}").status_code)
        on_progress(0.5)

    db, storage = Database(settings.db_url), LocalStorage(settings.data_dir)
    worker.run_once(db, storage, "test", engine)
    assert answers == [409]  # asked to stop; the run folder was still open
    assert client.delete(f"/matches/{mid}").status_code == 204
    assert client.get(f"/matches/{mid}").status_code == 404
    assert not storage.match_dir(mid).exists()


def test_a_job_left_behind_by_a_closed_app_can_be_stopped_and_deleted(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    mid = client.post("/matches", json={"title": "x"}).json()["id"]
    client.put(f"/matches/{mid}/video", content=data)
    client.post(f"/matches/{mid}/analyze")
    db = Database(settings.db_url)
    with db.sessions() as s:
        job = jobs.claim_next(s, "closed app")
        job.match.status = "processing"
        job.heartbeat_at = job.heartbeat_at - timedelta(seconds=settings.stale_job_s + 1)
        s.commit()
    # Nobody is left to see a stop request, so it stops straight away.
    assert client.post(f"/matches/{mid}/stop").json()["status"] == "stopped"
    with db.sessions() as s:
        assert jobs.requeue_stale(s, settings.stale_job_s) == 0  # and isn't started again
    client.post(f"/matches/{mid}/analyze")
    with db.sessions() as s:
        job = jobs.claim_next(s, "closed again")
        job.status = "stopping"  # the app closed while stopping
        job.heartbeat_at = job.heartbeat_at - timedelta(seconds=settings.stale_job_s + 1)
        s.commit()
        jobs.requeue_stale(s, settings.stale_job_s)
    assert client.get(f"/matches/{mid}").json()["status"] == "stopped"
    assert client.delete(f"/matches/{mid}").status_code == 204


def test_stopping_who_is_who_keeps_the_finished_results(client, settings, tmp_path):
    data = make_video(tmp_path / "v.mp4").read_bytes()
    mid = client.post("/matches", json={"title": "x"}).json()["id"]
    client.put(f"/matches/{mid}/video", content=data)
    client.put(f"/matches/{mid}/calibration", json={"image_size": [1280, 720], "points": COURT})
    client.post(f"/matches/{mid}/analyze")

    def engine(video, run_dir, calibration, config, on_progress, tags=()):
        (run_dir / "segments.json").write_text("{}")
        (run_dir / "summary.json").write_text(json.dumps({"names": {"P1": "Maya"}, "players": []}))

    db, storage = Database(settings.db_url), LocalStorage(settings.data_dir)
    assert worker.run_once(db, storage, "test", engine)

    def tag(t: float) -> dict:  # on a frame, so who is who is decided again
        body = {"time_s": t, "x": 700, "y": 600, "name": "Ali", "player": None}
        return client.post(f"/matches/{mid}/tags", json=body).json()

    # Queued: the match goes back to its results, not to "stopped" with nothing shown.
    assert tag(600)["job"]["kind"] == "relabel"
    m = client.post(f"/matches/{mid}/stop").json()
    assert m["status"] == "done" and m["job"]["status"] == "stopped"

    # Running: the same once the worker has stopped.
    assert tag(700)["job"]["status"] == "queued"

    def relabel(video, run_dir, calibration, config, on_progress, tags=()):
        on_progress(0.1)
        client.post(f"/matches/{mid}/stop")
        on_progress(0.5)

    assert worker.run_once(db, storage, "test", engine, relabel)
    m = client.get(f"/matches/{mid}").json()
    assert m["status"] == "done" and m["job"]["status"] == "stopped"
    assert client.get(f"/matches/{mid}/results").json()["tracking"]["names"] == {"P1": "Maya"}
