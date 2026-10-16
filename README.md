# Padel Buddy

Upload a long padel match video and get the rallies cut into clips, a
highlight reel, player movement stats and, later, shot and mistake analysis
with a written report.

It starts as a personal tool and is built so the same engine can serve a web
app, phones, an Android app and a desktop app.

## Status

| Step | What | State |
| --- | --- | --- |
| 1 | Court mapping and player tracking | In progress ([engine/](engine/)) |
| 2 | Rally detection, clips (±30 s around intense moments) and highlights | Next |
| 3 | API, job queue, storage and web app | Planned |
| 4 | Ball tracking, shots, mistakes and match report | Later |

See [docs/architecture.md](docs/architecture.md) for how the pieces fit.

## Quick start (step 1)

```bash
cd engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,calibrate,dev]"   # install PyTorch for your GPU first

padel-engine calibrate path/to/match.mp4 --out data/calibration.json
padel-engine track path/to/match.mp4 -c data/calibration.json --out runs/match1
```

Details in [engine/README.md](engine/README.md).

## Repository rules

This repository is public. Keys go in a local `.env` (git-ignored; copy
`.env.example`). Videos, run outputs and model weights are git-ignored too;
keep them under `data/` and `runs/`.
