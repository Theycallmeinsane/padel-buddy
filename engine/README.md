# padel-engine

The video engine behind Padel Buddy. It runs the same way from the command
line, from the API workers, and inside any future desktop app.

Current scope:

- **Step 1: court mapping and player tracking** (needs a calibration)
- **Step 2: rallies, clips and highlights** (from the sound; works without calibration)
- **Step 3: ball tracking and bounces** (during points; needs a calibration)
- **Step 4: shots, how each point ended, the match report and each player's
  highlights** (needs a calibration; better the more the ball is seen)

## Install

```bash
cd engine
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch for your machine (CUDA, Apple MPS or CPU): https://pytorch.org/get-started/locally/
# 2. The engine with detection and the calibration tool
pip install -e ".[detect,calibrate,dev]"
# 3. Optional: a written coach's summary in the report, from the Claude API
pip install -e ".[coach]"
```

## Use

```bash
# Once per camera position: click court keypoints on a frame.
padel-engine calibrate match.mp4 --out calibration.json

# Follow the players (any number: people may rotate in).
padel-engine track match.mp4 -c calibration.json --out runs/match1 --device cuda:0

# Track again without the detector (new settings, code or calibration), from what it saw last time.
padel-engine track match.mp4 -c calibration.json --out runs/again --detections runs/match1/detections.npz

# Clips and highlights from the sound (add --tracks runs/match1/tracks.csv to use movement too).
padel-engine clips match.mp4 --out runs/match1

# The ball, during the points clips found (uses the players' boxes from track).
padel-engine ball match.mp4 -c calibration.json --out runs/match1

# Shots, the match report and a highlight reel per player (after the steps above).
padel-engine report --out runs/match1 --video match.mp4

# Or everything in one go (tracking, the ball and the report only if -c is given).
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
| `tracks.csv` | One row per player per analysed frame: box, track id, track piece, court x/y in metres |
| `segments.json` | Every piece of track with its colours, times and places, so who is who can be decided again (after a tag) without tracking again |
| `detections.npz` | Everyone the detector saw in each analysed frame, with shirt colours (`output.detections`, tens of MB for a long match), so the match can be tracked again in minutes without the GPU |
| `summary.json` | Per-player distance, speed, time at net / mid / back, left / right share |
| `heatmaps/P1.png` … `all.png` | Top-down occupancy heatmaps |
| `preview.mp4` | With `track --preview` only: the video with player boxes and a minimap burnt in. The app's viewer draws the same live, so it is off by default |
| `overlay.json` | Court lines in the image and every player's box and smoothed court position, for the app's viewer |
| `ball.json` | How much of the points the ball was found in, flights and bounces |
| `ball_track.json` | The ball's position in every frame it was found, and each bounce in court metres |
| `hits.csv` | Every detected ball hit: time and strength |
| `clips.json` | Rallies (points), intense moments and clips with their times and scores |
| `clips/clip_001.mp4` … | The best points, each whole from 5 s before the serve to 3 s after the last shot, plus a `.jpg` thumbnail |
| `highlights.mp4` | The best rallies, cut tight (3 s either side) and joined in match order |
| `condensed.mp4` | Every point back to back with the dead time between points cut, at 720p |
| `points/point_001.mp4` … | Every point on its own (2 s either side), at 720p, numbered like the app's "Point 1" |
| `shots.json` | Every shot: time, point, who hit it, kind (serve, return, volley, smash, vibora, bandeja, lob, flat, groundstroke), where they stood, its pace and whether it stayed in play; and every point: shots, who played the last one, and how it ended |
| `report.json` | The match report: per team and per player numbers, formations, each player's success rate per kind of shot, tips, each player's best moments and winners, the shorts, and the coach's summary if enabled |
| `report.md` | The same report as a readable page |
| `players/P1.mp4` … | Each player's own highlight reel (their winners, overheads and long points they won), at 720p |
| `players/P1_winners.mp4` … | Every point the player finished with a winner, joined, at 720p |
| `shorts/point_012.mp4`, `shorts/P1_winner.mp4` … | Vertical 9:16 shorts for phones: the 5 best points, and each player's best winner |
| `live.json` | With a calibration: how far the results published so far reach (`covered_s` of `duration_s`), and `done` once the run finished |

Players are numbered **P1, P2, ...** in the order they first appear (at
the start, near-left first), however many there are: people may rotate in
and swap partners. Anyone on court for only a few seconds in all (fetching
a ball, a spectator by the door) is not a player, and their rows in
`tracks.csv` say `X`. A **team** is whoever shares a half during a point:
team A plays it on the camera's side of the net, team B on the far side,
and each point in `shots.json` lists who was on each side.

Court coordinates: `x` 0–10 m from the left wall to the right wall as seen
from the camera, `y` 0–20 m from the near back wall to the far one, net at
`y = 10`. Run `padel-engine keypoints` to list the named calibration points.

## How it works

1. **Calibration** fits a homography from clicked keypoints to the FIP court
   model (10 × 20 m, service lines 6.95 m from the net), so a tilted or
   off-centre camera is fine.
2. **Detection** runs YOLO (person class) with ByteTrack every
   `1 / process_fps` seconds (10 fps by default). Frames are decoded and
   shrunk to the detector's size in a background thread, and sent to the GPU
   8 at a time in half precision; ByteTrack still sees them one by one.
3. **Following people** projects each person's feet onto the court, merges
   two boxes on one person, and keeps four live slots (one per person on
   court) by matching on
   court position (gated by a maximum running speed), shirt colour, and the
   tracker's id. People watching stand just outside the glass, by the net
   posts and the side fences, so only someone inside the lines can become a
   player, a player is followed off court only while the tracker keeps the
   same id, and one followed outside the lines for 4 s is let go. The
   doorways by the net posts count as off court (0.6 m in from the side
   lines, 1.5 m either side of the net): people waiting to play, fetching
   balls or watching stand in them, often just inside the lines.
   When a player disappears (out of frame, behind another player, or just
   missed by the detector), nobody else's number changes and their own
   number waits for them; they simply have no box while they are gone. A
   player seen a moment ago always wins over one who has vanished, nobody is
   moved onto someone who clearly looks different unless the tracker says it
   is still the same person, and the tracker's id only counts when the
   colours agree, because it often hands a hidden player's id to whoever is
   in front of them. A player missing for a couple of seconds is re-found by
   shirt colour, and only by someone whose colours are close; that limit
   relaxes over 30 s in case their look changed. Each slot's track is cut
   into pieces wherever its person vanished for more than half a second or
   jumped, where two people part after passing within a metre of each
   other, since that is where a swap would happen, and where someone walks
   round the net to the other side, so each piece is seen from one side.

   **Who is who** is decided once the whole match is tracked
   (`identity.py`), for any number of people. Pieces seen at the same moment
   are different people. Pieces whose colours (shirt, shorts, legs) are
   alike are the same person, the most alike first, until nobody left is
   alike enough (`tracking.same_person_max_distance`); a piece that starts a
   moment after another ended, a step away, or that continues a track cut
   at a close pass, leans to being the same person. Colours are compared
   side by side of the net: from the far side everyone looks smaller,
   dimmer and is behind the net, so one person's near and far colours
   differ more than two people's on one side, and pieces only ever seen
   from opposite sides must look more alike (`tracking.cross_side_penalty`).
   One person's near and far pieces come together where the tracker
   followed them round the net with nobody close by: those pieces are the
   same person. Short pieces that match
   nobody well enough join the player they look most like. So a mix-up the
   live pass made is put right by colour, someone who rotates in becomes a
   new player, and someone who sat out gets their number back.

   **Names.** In the app you can point at a player at any moment and type
   their name (a tag), or type names in the report. A tag pins the piece of
   track under the click to that name: pieces with one name are one person
   and pieces with different names are different people, so naming two
   lookalikes tells them apart, and one tag after a swap at a close pass
   puts the rest of the match right. Naming again only re-runs who is who,
   shots and report from `segments.json` and `tracks.csv`, which takes
   seconds. A tag that can't hold (nobody tracked under it, or its name on
   someone else at that same moment) is listed in `summary.json` as not
   applied.
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
12. **Shots.** Every hit sound inside a point is a candidate shot. The hitter
   is the player whose box the ball is nearest at that moment (within 0.6 of
   their height); a hit with the ball in view but nobody near it was the
   glass or the floor and is dropped. When the ball isn't seen, the team is
   worked out from the nearest seen shot, since teams alternate. The kind
   comes from the shot's number (serve, return), where the ball was against
   the player (above their head: an overhead, a smash if at the net), where
   they stood (a volley at the net), where the
   ball landed next (a lob lands deep on the other side after a long
   flight) and the shot's pace: court metres from the hitter to the bounce
   per second in the air. An overhead from the back at 15 m/s or more is a
   vibora (22 m/s or more, a smash), slower a bandeja; a groundstroke at
   12 m/s or more is a flat drive. Without a bounce seen the pace is
   unknown, and the slower kind is kept. Forehand and backhand need a pose
   model and come later.
13. **How a point ended.** A bounce on the other side after the last shot is
   a winner. A last shot seen leaving that never lands on the other side is
   an error: "net" if it ends in the band around the net, otherwise "out".
   Anything else is left unknown rather than guessed.
14. **Match report.** Per player: points played and won, who they played
   with, shots by kind, winners, errors (and which shot they came from),
   time at the net, metres per point and top speed, counting only time
   during points. Per pair (two people who played at least 3 points
   together on one side): points played and won, and how often both were at
   the net, split, or both back. Tips are simple rules on those numbers (for
   example, a pair that is split a lot is open through the middle) and
   appear once there are at least 5 points.

   Per player and kind of shot, the **success rate** is the share kept in
   play (the other team had to play it back) or won outright, among shots
   whose outcome is known. With 5 or more such shots a kind is rated
   strong (90% or more), weak (under 75%) or solid, and each player's
   weakest kind becomes a tip with one thing to try.
15. **Player highlights.** Each player's winners, smashes, bandejas and long
   points their side won (8+ shots, 3+ of them theirs) are ranked, overlaps
   dropped, and the best 10 cut into `players/P#.mp4`. Every point the
   player finished with a winner (from 8 s before their shot to 3 s after)
   is also joined into `players/P#_winners.mp4`.
16. **Shorts**: the 5 best points, and each player's best winner, as
   vertical 9:16 clips of at most 59 s. A camera behind the court films in
   landscape, so instead of cropping players off, the frame is cut to the
   court's width and laid over a blurred copy of itself.
17. **Coach's summary** (optional). With the `coach` extra installed and
   `ANTHROPIC_API_KEY` set, the report's numbers (no video, no names, players
   as P1, P2, ...) are sent to Claude (`report.ai_model`, Claude Opus 5.5 by
   default) for a short written summary. Without a key this step is skipped.

18. **Live results** (with a calibration; `live.enabled`, on by default).
   Instead of everything at the end, results are published as the match is
   analysed. The sound is analysed first, so every point is listed within a
   minute or two. Tracking then publishes the stats, heatmaps and overlay so
   far after every 5 minutes of video (`live.every_s`), with who is who as
   it stands, tags included. Each time, the ball is tracked in the points
   tracking has passed, in a background thread while the GPU tracks on, and
   shots, point endings and the report's numbers follow for every point
   whose ball is done. Every point's own clip is cut in the background from
   the start; the condensed match at the end joins them. The best clips,
   highlight reel, player reels, shorts and the coach's summary come at the
   end, when every point can be ranked and who is who has seen the whole
   match, so numbers and even player numbers can still change then. The
   finished outputs are exactly those of a run without live results.
   Smoothing keeps the stretches of track that have not changed, so each
   publish costs a few seconds. Every output is written beside its name and
   moved over it in one step, so the app can read results at any moment,
   and if a run fails, what it published stays.

Settings live in `padel_engine/config.py` (the `audio`, `clips`, `ball`,
`shots` and `report` sections cover all of the above); pass a JSON file of
overrides with `--config`.
For example, to rule out a neighbouring court seen through the glass (pixels
of the image you calibrated on):

```json
{"ball": {"ignore_regions": [[[0, 540], [690, 540], [705, 633], [0, 930]]]}}
```

## Speed

Player tracking is the longest stage, because the detector runs on 10
frames of every second; encoding clips and reels and tracking the ball come
next. These settings are on by default and change no results:

| Setting | Default | What it does |
| --- | --- | --- |
| `detector.batch` | 8 | Frames sent to the GPU together; ByteTrack still updates frame by frame |
| `detector.half` | true | FP16 inference on an NVIDIA GPU (ignored on the CPU) |
| `tracking.prefetch_frames` | 16 | Frames decoded and shrunk ahead in a background thread while the GPU works |
| `ball.workers` | 0 (auto) | Points tracked at the same time; auto is a quarter of the CPU threads, at most 4 |
| `clips.video_encoder` | auto | NVENC (`h264_nvenc`) when the GPU and FFmpeg support it, else x264 |
| `live.enabled` | true | Point clips and the ball are worked on while the GPU tracks, instead of after it |

These trade accuracy for more speed, so test them on your own footage first:
`tracking.process_fps` (10), `detector.image_size` (1280; 960 is about 1.5
times faster on an RTX 3050 but may miss small far-side players) and
`detector.model` (`yolo11n.pt` is faster, `yolo11m.pt` slower and more
reliable).

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
  Teammates in identical kit can swap ids at that moment. A name tag after
  the swap fixes it.
- A tag fixes a whole piece of track, and pieces end where a player
  vanishes or passes close to another. Two tags on the same piece that
  disagree can't both hold; the later one wins and the earlier one is shown
  as not applied.
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
- Who hit each shot and its kind are only as good as the ball tracking. With
  the ball unseen at a hit, the team is a guess from the order of play
  (counted from the nearest hit where the ball was seen), so a missed hit
  sound flips the guess until the ball is seen again. Shot
  kinds and the reach, lob and pace thresholds (`shots` settings) need
  tuning on real matches; success rates are only as good as the endings
  behind them.
