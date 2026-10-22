# Architecture

One Python engine does all video work. Every client (web, phone, Android,
desktop) talks to it through the same API, so adding a client never means
re-implementing analysis.

```
 clients                     backend                         engine (Python)
 ───────                     ───────                         ───────────────
 web app (Next.js, PWA) ─┐
 Android app ────────────┼──► API (FastAPI) ──► job queue ──► workers ──► padel_engine
 desktop app ────────────┘        │                              │
 CLI on your machine ─────────────┼──────────────────────────────┘ (direct, no API)
                                  ▼
                       Postgres + object storage (Supabase)
                       matches, runs, clips, stats, reports
```

## Engine (`engine/`)

A plain library with a CLI on top. Each stage reads the previous stage's
output files, so stages can be re-run independently and the API can expose
them one by one.

| Module | Role |
| --- | --- |
| `court` | FIP court model in metres: keypoints, lines, depth zones |
| `calibration` | Image ↔ court homography from clicked keypoints |
| `detection` | `PersonTracker` interface; YOLO + ByteTrack implementation |
| `players` | Four stable player ids from noisy detections, and the repair that fixes swaps after the match is tracked |
| `tags` | Names someone gave by clicking a player: which track piece each lands on, and which number each name takes |
| `analytics` | Kalman smoothing, distance, speed, zones, heatmaps |
| `render` | Calibration overlay, annotated preview, top-down court |
| `pipeline` | `run_tracking(...)`, and `run_match(...)`: tracking then clips, the single entry point; `apply_tags(...)` re-decides who is who in a finished run |
| `audio` | Streams audio through FFmpeg; ball-hit onsets and loudness |
| `events` | Rallies, intense moments, clip windows and the highlight selection |
| `clips` | `run_clips(...)`: cuts clips, thumbnails and the highlight reel with FFmpeg |
| `ball` | `run_ball(...)`: ball candidates from motion, flights and bounces during points |
| `shots` | `run_shots(...)`: who hit each shot and its kind, and how each point ended |
| `report` | `run_report(...)`: team and player numbers, success rates per shot, tips, each player's moments, reels and winners reel |
| `shorts` | Vertical 9:16 clips of single points for phones, keeping the whole court in view |
| `live` | Publishes results while a match is analysed: points first, then stats, ball, shots and report every few minutes |
| `files` | Writes each output beside its name and moves it over in one step, so readers never see half a file |
| `seen` | Saves everyone the detector saw (`detections.npz`) and replays it, to track a match again without the GPU |
| `coach` | Optional written summary of the report's numbers from the Claude API |
| `schemas` | Pydantic output models, the contract the API will serve |

Design choices:

- **Court coordinates everywhere.** Everything after detection works in
  metres on the court, not pixels. That keeps stats physical and makes a
  new camera angle just a new calibration.
- **Swappable models.** The pipeline only depends on the `PersonTracker`
  protocol, so a stronger detector or a fine-tuned padel model drops in
  without touching the rest.
- **Uncertainty-aware smoothing.** A pixel of error is centimetres near the
  camera but tens of centimetres at the far wall. The homography's Jacobian
  turns pixel jitter into per-point uncertainty, and a Kalman smoother uses
  it, so far-side distances are not inflated by noise.
- **Keep the GPU busy.** Tracking decodes and shrinks frames in a background
  thread and sends them to the detector 8 at a time in half precision; the
  ball tracker works on several points at once; clips are encoded with NVENC
  when there is an NVIDIA GPU. None of it changes results (see "Speed" in
  [engine/README.md](../engine/README.md)).
- **Run folders.** Every run writes to one folder (`tracks.csv`,
  `summary.json`, images, preview). Locally that is a directory; in the
  cloud the worker uploads the same folder to storage. With live results the
  folder fills while the match is analysed, and `live.json` says how far.
- **Detect once.** Finding people is most of tracking's time. The run
  keeps what the detector saw (`detections.npz`), so a change to who-is-who
  logic, settings or the calibration tracks the match again in minutes,
  with exactly the same people as input.

## API (`api/`)

FastAPI plus a worker process. Uploads are resumable (chunks with
`Content-Range`), the jobs table doubles as the queue (workers claim rows
with a conditional `UPDATE`), and SQLite or Postgres is just a URL. Details
in [api/README.md](../api/README.md).

## Web app (`web/`)

A static Next.js export: upload with pause and resume, in-browser court
calibration, naming players by clicking them, progress, the match viewer, the match report, each player's
highlights, a page per player across matches, the highlight reel, clip
gallery, player table and heatmaps. Being static, the same build can be wrapped as the Android app
(Capacitor) and the desktop app (Tauri) later without a server of its own.

## Roadmap

1. **Court mapping and player tracking.** Built.
2. **Clips and highlights.** Built: intense moments from ball-hit audio
   (ranked with player movement when tracking ran), clips from 30 s before
   to 30 s after, overlapping windows merged so a long rally is one clip,
   and a reel of the best rallies.
3. **API and web app.** Built. Next for this step: object storage for a
   cloud deployment, and the Android and desktop shells.
4. **Shots and mistakes.** Built: ball tracking and bounces, who hit each
   shot and its kind, how each point ended (winner, net, out), a match
   report with tips and an optional coach's summary, and each player's own
   highlights and numbers, followed across matches by name. Players are
   named by clicking them on the video; a tag after a tracking mix-up
   re-runs only the identity repair, shots and report.
5. **Pose.** Forehand and backhand, technique notes, and a trained ball
   detector, once real footage is available to train and tune on.

Automatic court-line detection (no clicking) is planned once real footage is
available to tune it on.
