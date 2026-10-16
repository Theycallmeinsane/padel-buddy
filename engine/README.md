# padel-engine

The video engine behind Padel Buddy. It runs the same way from the command
line, from the API workers, and inside any future desktop app.

Current scope (step 1): **court mapping and player tracking**.

## Install

```bash
cd engine
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch for your machine (CUDA, Apple MPS or CPU): https://pytorch.org/get-started/locally/
# 2. The engine with detection and the calibration tool
pip install -e ".[detect,calibrate,dev]"
```

## Use

```bash
# Once per camera position: click court keypoints on a frame.
padel-engine calibrate match.mp4 --out calibration.json

# Track the four players.
padel-engine track match.mp4 -c calibration.json --out runs/match1 --device cuda:0
```

`calibrate` shows the frame beside a court diagram and asks for one keypoint
at a time. Click it, or press Enter to skip a point that is hidden or out of
shot. Four points is the minimum; the far corners plus both ends of both
service lines (six points) is a good set for a camera behind the court. It
also writes `calibration.check.jpg` with the fitted court drawn over the
frame, so you can confirm the yellow lines sit on the real ones.

The calibration is reused for every video from the same camera position,
and it scales automatically if you record at another resolution.

## Outputs

Each run folder contains:

| File | What it is |
| --- | --- |
| `calibration_check.jpg` | The court model drawn over the first frame |
| `tracks.csv` | One row per player per analysed frame: box, track id, court x/y in metres |
| `summary.json` | Per-player distance, speed, time at net / mid / back, left / right share |
| `heatmaps/P1.png` … `all.png` | Top-down occupancy heatmaps |
| `preview.mp4` | The video with player boxes and a live top-down minimap |

Players are numbered by where they stand when first seen: **P1** near-left
and **P2** near-right (team A, camera side at the start), **P3** far-left
and **P4** far-right (team B). Ids follow the person through side switches.

Court coordinates: `x` 0–10 m from the left wall to the right wall as seen
from the camera, `y` 0–20 m from the near back wall to the far one, net at
`y = 10`. Run `padel-engine keypoints` to list the named calibration points.

## How it works

1. **Calibration** fits a homography from clicked keypoints to the FIP court
   model (10 × 20 m, service lines 6.95 m from the net), so a tilted or
   off-centre camera is fine.
2. **Detection** runs YOLO (person class) with ByteTrack every
   `1 / process_fps` seconds (10 fps by default).
3. **Player assignment** projects each person's feet onto the court, drops
   anyone outside the court (spectators, neighbouring courts), and keeps four
   stable player slots by matching on court position (gated by a maximum
   running speed), shirt colour, and the tracker's id. A player missing for a
   couple of seconds is re-found by shirt colour.
4. **Analytics** smooths positions with a Kalman smoother that knows a
   pixel of jitter is a few centimetres near the camera but tens of
   centimetres at the far wall, then computes distance, speed, depth zones
   and heatmaps.

Settings live in `padel_engine/config.py`; pass a JSON file of overrides with
`--config`.

## Tests

```bash
pytest
```

The tests use a synthetic camera and scripted players, so they need neither
a GPU nor real footage.

## Known limits (to tune with real footage)

- Far-side players' feet can be hidden by the net from a low camera; their
  position then comes from YOLO's box bottom, which may sit a little high.
  Use a higher camera stand or a larger model (`--model yolo11m.pt`) if far
  players drift.
- At changeovers, players who leave the frame are re-found by shirt colour.
  Teammates in identical kit can swap ids at that moment.
- Calibration is manual for now; automatic court-line detection comes once
  we have sample footage to tune it on.
