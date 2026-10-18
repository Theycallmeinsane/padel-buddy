# padel-engine

The video engine behind Padel Buddy. It runs the same way from the command
line, from the API workers, and inside any future desktop app.

Current scope:

- **Step 1: court mapping and player tracking** (needs a calibration)
- **Step 2: rallies, clips and highlights** (from the sound; works without calibration)
- **Step 3: ball tracking and bounces** (during points; needs a calibration)

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

# The ball, during the points clips found (uses the players' boxes from track).
padel-engine ball match.mp4 -c calibration.json --out runs/match1

# Or everything in one go (tracking and the ball only if -c is given).
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
| `overlay.json` | Court lines in the image and every player's box and smoothed court position, for the app's viewer |
| `ball.json` | How much of the points the ball was found in, flights and bounces |
| `ball_track.json` | The ball's position in every frame it was found, and each bounce in court metres |
| `hits.csv` | Every detected ball hit: time and strength |
| `clips.json` | Rallies (points), intense moments and clips with their times and scores |
| `clips/clip_001.mp4` … | The best points, each whole from 5 s before the serve to 3 s after the last shot, plus a `.jpg` thumbnail |
| `highlights.mp4` | The best rallies, cut tight (3 s either side) and joined in match order |
| `condensed.mp4` | Every point back to back with the dead time between points cut, at 720p |

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
6. **Rallies** are runs of at least 4 hits with no gap longer than 3.5 s;
   each one is a point. Each rally is scored against the others in the
   match on length, number of shots and (if tracking ran) how hard the
   players moved. Loudness is left out on purpose: a phone behind one glass
   hears balls hitting that glass far louder than anything else, so loud
   means "near the camera", not "good point".
7. **Clips** are the best points, whole: the top 8 per hour of video (at
   least 3), each from 5 s before the serve to 3 s after the last shot, so
   the build-up and the finish are both in. Points close enough to overlap
   share a clip.
8. **Highlight reel**: the 10 best rallies, cut tight and joined.
9. **Condensed match**: every rally with 2 s either side, joined, so the
   whole match plays without the walking, ball fetching and towel breaks.
10. **Intense moments** (`--mode moments`, the old rule) are sounds at least
   6 dB louder than the match's typical shot, during a rally or within 4 s
   after it. Each rally keeps its loudest, the top 8 per hour are kept, and
   clips run from 30 s before to 30 s after each one, widened so a rally is
   never cut mid-point. They are still listed in `clips.json` in points mode.

11. **Ball** (with a calibration), during each point plus a second either
   side, on frames shrunk to 960 px wide. A pixel is moving when it differs
   from both the previous and the next frame. Pixels that move most of the
   time (lights, fences), the players' boxes, anything below the court floor
   as the camera sees it, and any `ball.ignore_regions` are masked out.
   Small compact moving spots are linked frame to frame with a
   constant-velocity guess into flights; a flight must travel and move
   smoothly, which drops arms and rackets. Where flights claim the same
   frame the longest wins. A **bounce** is where a flight is lowest in the
   image and that spot is on the court: the down and up paths are crossed to
   find the contact between frames, because the nearest frame has the ball
   a little up, which at the far wall moves the bounce most of a metre.

Settings live in `padel_engine/config.py` (the `audio`, `clips` and `ball`
sections cover all of the above); pass a JSON file of overrides with `--config`.
For example, to rule out a neighbouring court seen through the glass (pixels
of the image you calibrated on):

```json
{"ball": {"ignore_regions": [[[0, 540], [690, 540], [705, 633], [0, 930]]]}}
```

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
- The ball tracker has no model of what a ball looks like, only of how it
  moves. A ball in front of a busy background, or very small at the far
  wall, is missed, and a fast racket can pass for a short flight. A trained
  detector (TrackNet, fine-tuned on padel frames) is the next step; it
  replaces only the candidate stage.
- Motion seen through the glass (the next court, people walking past) can
  look like a ball. Rule those areas out with `ball.ignore_regions`.
- Hit sensitivity and the "intense" bar are tuned on synthetic audio and
  need checking against real recordings and phone microphones.
