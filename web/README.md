# Padel Buddy web app

Upload a match, optionally click the court once, run the analysis and watch
the clips, highlight reel, player stats and heatmaps. Works on desktop and
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
- **Match** (`/match/?id=…`): calibration, analysis progress, highlight reel,
  clip gallery, player table and heatmaps.
- **Settings** (`/settings/`): API address and optional token.

## Checks

```bash
npm test           # upload/resume logic and the API client
npm run typecheck
npm run build
```
