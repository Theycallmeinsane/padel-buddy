# padel-api

The HTTP API and job worker behind Padel Buddy. The web app (and later the
Android and desktop apps) talk to this; the worker runs the engine.

```
web / phone / desktop ──► padel-api ──► database (jobs table) ◄── padel-worker ──► padel_engine
                              │                                        │
                              └────────── data/matches/<id>/ ──────────┘
                                          video.mp4, run/ (clips, reel, stats)
```

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "./engine[detect]" -e "./api[dev]"     # from the repo root

padel-api            # http://127.0.0.1:8000, interactive docs at /docs
padel-worker         # in a second terminal; one per GPU
```

Everything lands in `./data` (SQLite database, videos, outputs), which is
git-ignored. Settings come from environment variables or `.env`; see
`.env.example` in the repo root.

## Endpoints

| Method | Path | What |
| --- | --- | --- |
| `POST` | `/matches` | Create a match: `{"title", "filename", "size"}` |
| `PUT` | `/matches/{id}/video` | Upload the video, whole or in chunks with `Content-Range: bytes a-b/total` |
| `GET` | `/matches/{id}/video/status` | Bytes received so far, to resume an interrupted upload |
| `GET` | `/matches/{id}/frame?t=5` | A still frame (JPEG) to click the court on |
| `PUT` | `/matches/{id}/calibration` | Clicked court points; enables player tracking |
| `POST` | `/matches/{id}/analyze` | Queue analysis, optionally with engine setting overrides |
| `GET` | `/matches/{id}` | Status and job progress |
| `GET` | `/matches/{id}/results` | Tracking and clips summaries (`summary.json`, `clips.json`) |
| `GET` | `/matches/{id}/files/{path}` | Any output: clips, reel, thumbnails, heatmaps, preview (supports Range for video seeking) |
| `GET` | `/matches/{id}/video` | The original upload |
| `GET` | `/court/keypoints` | Names and positions of the calibration points |
| `DELETE` | `/matches/{id}` | Remove a match and its files |

## Design notes

- **Resumable uploads without extra services.** Match videos are several
  gigabytes. Each chunk says where it starts; the server refuses gaps and
  reports how much it holds, so a client resumes after a dropped
  connection instead of starting again.
- **The database is the queue.** Workers claim jobs with a conditional
  `UPDATE`, which only one can win. No Redis or broker to run, it survives
  restarts, and it scales to several workers on Postgres. A job whose
  worker goes quiet for 10 minutes is put back in the queue.
- **SQLite locally, Postgres in the cloud.** Set
  `PADEL_DATABASE_URL=postgresql+psycopg://...` (for example a Supabase
  database) and install `padel-api[postgres]`.
- **Storage behind one class.** `storage.LocalStorage` decides where a
  match's files live. A cloud deployment either shares one disk between
  API and workers or swaps in an object-storage version of that class.
- **Auth.** For personal use, set `PADEL_API_TOKEN` and send it as a Bearer
  token (or `?token=` on media links). Per-user accounts come with the
  multi-user product.

## Tests

```bash
pytest
```

The suite uploads in chunks, resumes, calibrates, runs jobs with a fake
engine and once with the real engine on a synthetic match (needs FFmpeg).
