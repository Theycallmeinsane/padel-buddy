# Padel Buddy web app

Upload a match, optionally click the court once, run the analysis and watch
the clips, highlight reel, player stats and heatmaps, then read the match
report and each player's own highlights. Works on desktop and
phone browsers, and can be installed to a phone's home screen.

It is a static Next.js export (`out/`): plain files any host can serve. The
same build can later be wrapped as the Android app (Capacitor) and the
desktop app (Tauri), since all data comes from the API at runtime.

## Run it

Start the API and a worker first (see [../api/README.md](../api/README.md)), then:

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

The app talks to `http://localhost:8000` by default. Change it under
**Settings** (stored in your browser), or build with
`NEXT_PUBLIC_API_URL=https://your-api.example.com npm run build`.

To use it from your phone on the same Wi-Fi, run the API with
`padel-api --host 0.0.0.0`, add your computer's address to
`PADEL_CORS_ORIGINS`, and enter `http://<computer-ip>:8000` in Settings.

## Pages

- **Matches** (`/`): upload with pause and resume (pick the same file again
  after a reload and it continues where it stopped), and the list of matches.
- **Match** (`/match/?id=…`): calibration, naming the players, analysis progress, the viewer,
  the match report, each player's highlights, the highlight reel, best
  points, player table and heatmaps.

The viewer plays the condensed match (every point back to back) or the full
video, and draws over it: court lines, a box per player, the ball with a
short tail and a ring where it bounces, and a top-down minimap with the last
two seconds of movement and recent bounces. Below it, a timeline of every
point; the best ones are highlighted, and clicking one plays it. Lines, players
and the ball need a calibrated court; they come from the engine's `overlay.json`
and `ball_track.json`. With shots found, each box also shows the player's
name and the shot they just played, the status line says how the point ended,
and each point on the timeline is edged in the colour of the team that won it.
The original video only plays where the browser supports its codec (iPhone
HEVC often doesn't), so points-only is the default.

**Naming players.** After calibrating, **Name players** shows a frame: click
each player and type their name, and the names follow them through the whole
analysis. While watching, **Name a player** lets you click anyone at any
moment and say who it is. If tracking mixed two players up, tag the right
name after the mix-up and the rest of the match is put right in a few
seconds, shots and report included. Tags are listed under the viewer;
clicking one plays that moment, × takes it back, and a tag that couldn't be
applied is marked so you can tag that player at another moment.

  The **match report** below the viewer has the team and player tables,
  tips and, if a Claude key is set on the server, a coach's summary; it
  downloads as Markdown. **Name the players** there once per match. **Player
  highlights** picks one player: their numbers, their own highlight reel or
  (switch to **Winners**) every point they finished with a winner, a list of
  those moments (clicking one plays it in the viewer, which then dims
  everyone else), a table of each kind of shot with how often it stayed in
  play and a strong / solid / weak rating, and their vertical short.
  **Shorts** shows the best points as vertical clips for phones, and **Every
  point** lists all the points with how each ended, to play in the viewer or
  download as a clip of its own.
- **Players** (`/players/`): everyone named in a match. Each name opens
  their page (`/players/?name=…`) with totals and their card from every match.
- **Settings** (`/settings/`): API address and optional token.

## Checks

```bash
npm test           # upload/resume logic, the API client, the viewer's clock, tagging
npm run typecheck
npm run build
```
