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
| `players` | Four stable player ids from noisy detections |
| `analytics` | Kalman smoothing, distance, speed, zones, heatmaps |
| `render` | Calibration overlay, annotated preview, top-down court |
| `pipeline` | `run_tracking(video, calibration, out_dir)`: the single entry point |
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
- **Run folders.** Every run writes to one folder (`tracks.csv`,
  `summary.json`, images, preview). Locally that is a directory; in the
  cloud the worker uploads the same folder to storage.

## Roadmap

1. **Court mapping and player tracking** (this step).
2. **Clips and highlights.** Detect intense moments from ball-hit audio and
   player movement, cut clips from 30 s before to 30 s after, merging
   overlapping windows so a long rally is one clip, then rank highlights.
3. **API and web app.** FastAPI with background workers and resumable
   uploads; Supabase for Postgres and storage; a responsive Next.js app with
   in-browser calibration (click the court once per camera position).
4. **Shots and mistakes.** Ball tracking, pose, shot classification, error
   detection and a written match report built on the tracking data.

Automatic court-line detection (no clicking) is planned once real footage is
available to tune it on.
