# padel-engine

The video engine behind Padel Buddy. It runs the same way from the command
line, from the API workers, and inside any future desktop app.

Current scope:

- **Step 1: court mapping and player tracking** (needs a calibration)
- **Step 2: rallies, clips and highlights** (from the sound; works without calibration)

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

# Clips and highlights from the sound (add --tracks runs/match1/tracks.csv to use movement too).
padel-engine clips match.mp4 --out runs/match1

# Or everything in one go (tracking only if -c is given).
padel-engine analyze match.mp4 -c calibration.json --out runs/match1
```

FFmpeg must be installed for clips (`brew install ffmpeg`, `apt install ffmpeg`,
or `winget install ffmpeg`).

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
| `hits.csv` | Every detected ball hit: time and strength |
| `clips.json` | Rallies, intense moments and clips with their times and scores |
| `clips/clip_001.mp4` … | 30 s before to 30 s after each intense moment, plus a `.jpg` thumbnail |
| `highlights.mp4` | The best rallies, cut tight (3 s either side) and joined in match order |

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
5. **Hits** are sharp "pops" in the 1.5 to 8 kHz band, found as spectral-flux
   peaks above a threshold that adapts to the local noise. Voices and
   footsteps sit lower in frequency and are ignored. Audio is analysed a
   minute at a time, so a two-hour match takes seconds and little memory.
6. **Rallies** are runs of at least 4 hits with no gap longer than 3.5 s.
   Each rally is scored against the others in the match on length, number
   of shots, loudness and (if tracking ran) how hard the players moved.
7. **Intense moments** are sounds at least 6 dB louder than the match's
   typical shot, during a rally or within 4 s after it: smashes, wall
   bangs, cheers, shouts. Measuring against the match's own shots makes
   "intense" mean the same in a quiet session and a noisy club. On a real
   95-minute match almost every rally clears that bar, so it only makes a
   sound a candidate: each rally keeps its best one, and only the top 8 per
   hour of video (at least 3) are kept. If nothing clears the bar, the
   loudest instant of each top rally is used instead.
8. **Clips** run from 30 s before to 30 s after each moment, widened so a
   rally is never cut mid-point. Overlapping windows merge into one clip up
   to 3 minutes; after that a new clip starts (it may share a few seconds).
9. **Highlight reel**: the 10 best rallies, cut tight and joined.

Settings live in `padel_engine/config.py` (the `audio` and `clips` sections
cover all of the above); pass a JSON file of overrides with `--config`.

## Tests

```bash
pytest
```

The tests use a synthetic camera, scripted players and synthetic match audio,
so they need neither a GPU nor real footage. Clip tests need FFmpeg.

## Known limits (to tune with real footage)

- Far-side players' feet can be hidden by the net from a low camera; their
  position then comes from YOLO's box bottom, which may sit a little high.
  Use a higher camera stand or a larger model (`--model yolo11m.pt`) if far
  players drift.
- At changeovers, players who leave the frame are re-found by shirt colour.
  Teammates in identical kit can swap ids at that moment.
- Calibration is manual for now; automatic court-line detection comes once
  we have sample footage to tune it on.
- Hits from a neighbouring court sound the same as ours. Rallies there can
  show up as rallies here; player movement (from tracking) and, later, ball
  tracking will filter them.
- Hit sensitivity and the "intense" bar are tuned on synthetic audio and
  need checking against real recordings and phone microphones.
