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

Or run everything in one process on your own computer, with the web app at
http://127.0.0.1:8000 (build it first with `npm run build` in `web/`):

```bash
padel-buddy                                  # API + worker + web app, opens the browser
padel-buddy import ~/Videos/match.mov --analyze   # add a video already on disk
```

`import` hard-links the video into the data folder when it is on the same
drive, so a large match takes no extra space and deleting the match never
touches the original.

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
| `GET` | `/matches/{id}/results` | Every summary at once: tracking, clips, ball, shots and the match report |
| `GET` | `/matches/{id}/report.md` | The match report as Markdown, with the players' names |
| `PUT` | `/matches/{id}/players` | Name the players: `{"names": {"P1": "Maya", "P3": "Sara"}}`; an empty name clears one |
| `GET` | `/matches/{id}/players/{P1-P4}` | One player in one match: their numbers, shots, best moments and own highlight reel |
| `GET` | `/people` | Everyone named in any match, with how many matches each |
| `GET` | `/people/{name}` | One person across all their matches, newest first (name match ignores case) |
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
- **People are names.** A match stores which name played as P1-P4, so the
  same name in several matches is one person. No accounts needed; when
  accounts arrive, a name can be linked to one.
- **Claude key stays on the server.** Put `ANTHROPIC_API_KEY` in `.env` for
  the written coach's summary; the worker passes it to the engine and it
  never reaches the browser. Without it, reports just skip the summary.
- **Auth.** For personal use, set `PADEL_API_TOKEN` and send it as a Bearer
  token (or `?token=` on media links). Per-user accounts come with the
  multi-user product.

## Tests

```bash
pytest
```

The suite uploads in chunks, resumes, calibrates, runs jobs with a fake
engine and once with the real engine on a synthetic match (needs FFmpeg).
