"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { Api, Overlay, Rally, Results } from "@/lib/api";
import { lowerBound, nearest, toMatchTime, toVideoTime, type Segment } from "@/lib/timeline";
import { formatTime } from "./common";

// Same colours as the engine's preview video: team A in blues, team B warm.
const PLAYER_COLORS: Record<string, string> = { P1: "#1e8ceb", P2: "#28dce6", P3: "#f58c28", P4: "#e63c3c" };
const BOX_GAP_S = 0.3; // hide a player when their nearest sample is further away than this
const TRAIL_S = 2;

// Painted lines in court metres (x across, y from the near wall), net at y = 10.
const COURT_LINES: [number, number, number, number][] = [
  [0, 3.05, 10, 3.05],
  [0, 16.95, 10, 16.95],
  [5, 2.85, 5, 17.15],
];

type Layers = { lines: boolean; players: boolean; minimap: boolean };
type Source = "points" | "full";

export default function MatchViewer({
  api,
  matchId,
  durationS,
  data,
}: {
  api: Api;
  matchId: string;
  durationS: number | null;
  data: Results;
}) {
  const clips = data.clips;
  const rallies = useMemo(() => clips?.rallies ?? [], [clips]);
  const overlayFile = data.tracking?.outputs.overlay ?? null;
  const condensed = clips?.condensed && clips.condensed_segments?.length ? clips.condensed : null;
  const total = durationS || Math.max(1, ...rallies.map((r) => r.end_s));
  const best = useMemo(() => new Set(clips?.clips.flatMap((c) => c.rallies ?? []) ?? []), [clips]);

  const [source, setSource] = useState<Source>(condensed ? "points" : "full");
  const [fullFailed, setFullFailed] = useState(false);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [layers, setLayers] = useState<Layers>({ lines: true, players: true, minimap: true });
  const [matchTime, setMatchTime] = useState(0);
  const narrow = useMediaQuery("(max-width: 720px)");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resumeAt = useRef<number | null>(null);

  const segments = useMemo<Segment[] | null>(
    () => (source === "points" && condensed ? (clips?.condensed_segments as Segment[]) : null),
    [source, condensed, clips],
  );

  useEffect(() => {
    if (!overlayFile) return;
    let live = true;
    api.overlay(matchId, overlayFile).then(
      (o) => live && setOverlay(o),
      (e) => live && setOverlayError((e as Error).message),
    );
    return () => {
      live = false;
    };
  }, [api, matchId, overlayFile]);

  // Repaint every frame: it is cheap and follows playback, seeking and resizing alike.
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    let raf = 0;
    let shown = -1;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const t = toMatchTime(segments, video.currentTime);
      // React only needs ~10 updates a second, for the minimap and the timeline.
      if (Math.abs(t - shown) >= 0.1) {
        shown = t;
        setMatchTime(t);
      }
      paint(canvas, video, overlay, layers, t);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [overlay, layers, segments]);

  const seek = useCallback(
    (t: number) => {
      const video = videoRef.current;
      const at = toVideoTime(segments, Math.max(0, t));
      if (!video || at === null) return;
      video.currentTime = at;
      video.play().catch(() => {});
    },
    [segments],
  );

  function switchTo(next: Source) {
    resumeAt.current = matchTime;
    setSource(next);
  }

  const point = rallies.find((r) => r.start_s - 2 <= matchTime && matchTime <= r.end_s + 2);
  const src = source === "points" && condensed ? api.fileUrl(matchId, condensed) : api.videoUrl(matchId);

  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 12 }}>
        <h2 className="grow" style={{ margin: 0 }}>
          Watch
        </h2>
        {condensed && (
          <div className="segmented" role="group" aria-label="What to play">
            <button className={source === "points" ? "on" : ""} onClick={() => switchTo("points")}>
              Points only{clips?.condensed_s ? ` · ${formatTime(clips.condensed_s)}` : ""}
            </button>
            <button className={source === "full" ? "on" : ""} onClick={() => switchTo("full")} disabled={fullFailed}>
              Full match
            </button>
          </div>
        )}
      </div>

      <div className={`viewer${overlay && layers.minimap ? " with-map" : ""}`}>
        <div className="stage">
          <video
            ref={videoRef}
            src={src}
            controls
            playsInline
            preload="metadata"
            onLoadedMetadata={(e) => {
              if (resumeAt.current === null) return;
              const at = toVideoTime(segments, resumeAt.current);
              if (at !== null) e.currentTarget.currentTime = at;
              resumeAt.current = null;
            }}
            onError={() => {
              if (source !== "full") return;
              setFullFailed(true);
              if (condensed) setSource("points");
            }}
          />
          <canvas ref={canvasRef} aria-hidden="true" />
        </div>
        {overlay && layers.minimap && <Minimap overlay={overlay} t={matchTime} sideways={narrow} />}
      </div>

      <div className="row small" style={{ marginTop: 8 }}>
        <span className="grow muted">
          {formatTime(matchTime)}
          {point
            ? ` · point ${point.index + 1} of ${rallies.length} · ${point.hits} shots${best.has(point.index) ? " · one of the best" : ""}`
            : ""}
        </span>
        {overlay &&
          (
            [
              ["lines", "Court lines"],
              ["players", "Players"],
              ["minimap", "Minimap"],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="toggle">
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={(e) => setLayers({ ...layers, [key]: e.target.checked })}
              />
              {label}
            </label>
          ))}
      </div>

      {rallies.length > 0 && <Timeline rallies={rallies} best={best} total={total} t={matchTime} onSeek={seek} />}

      {fullFailed && (
        <p className="muted small">
          This browser can&apos;t play the original video (iPhone recordings are often HEVC).{" "}
          {condensed ? "Points only plays everywhere." : "Analyse again to get a condensed match that plays everywhere."}
        </p>
      )}
      {!overlayFile && (
        <p className="muted small">Calibrate the court and analyse again to see court lines and the players.</p>
      )}
      {overlayError && <p className="error small">Couldn&apos;t load the tracking overlay: {overlayError}</p>}
    </section>
  );
}

function paint(canvas: HTMLCanvasElement, video: HTMLVideoElement, overlay: Overlay | null, layers: Layers, t: number) {
  const dpr = window.devicePixelRatio || 1;
  const w = video.clientWidth;
  const h = video.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!overlay || !video.videoWidth || !(layers.lines || layers.players)) return;

  // The picture is letterboxed inside the element, and may be smaller than the
  // source the engine measured in (the condensed match is 720p).
  const fit = Math.min(w / video.videoWidth, h / video.videoHeight);
  const pw = video.videoWidth * fit;
  const ph = video.videoHeight * fit;
  const s = pw / overlay.size[0];
  ctx.setTransform(dpr * s, 0, 0, dpr * s, (dpr * (w - pw)) / 2, (dpr * (h - ph)) / 2);
  const px = 1 / s; // one screen pixel, in source pixels

  if (layers.lines) {
    ctx.lineWidth = 1.5 * px;
    for (const line of overlay.lines) {
      ctx.strokeStyle = line.name === "net" ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.65)";
      ctx.beginPath();
      line.points.forEach(([u, v], i) => (i ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
      ctx.stroke();
    }
  }
  if (layers.players) {
    ctx.lineWidth = 2 * px;
    ctx.font = `600 ${13 * px}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    for (const [pid, p] of Object.entries(overlay.players)) {
      const i = nearest(p.t, t, BOX_GAP_S);
      if (i < 0) continue;
      const [x1, y1, x2, y2] = p.box.slice(4 * i, 4 * i + 4);
      ctx.strokeStyle = ctx.fillStyle = PLAYER_COLORS[pid] ?? "#ffffff";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(pid, x1, y1 - 3 * px);
    }
  }
}

function Minimap({ overlay, t, sideways }: { overlay: Overlay; t: number; sideways: boolean }) {
  // Upright: far wall at the top, as the camera sees it. Sideways (phones, under
  // the video): turned a quarter clockwise, far wall on the right.
  const P = (x: number, y: number): [number, number] => (sideways ? [y, x] : [x, 20 - y]);
  const line = (x1: number, y1: number, x2: number, y2: number) => {
    const [a, b] = P(x1, y1);
    const [c, d] = P(x2, y2);
    return { x1: a, y1: b, x2: c, y2: d };
  };
  return (
    <svg
      className="minimap"
      viewBox={sideways ? "-0.8 -0.8 21.6 11.6" : "-0.8 -0.8 11.6 21.6"}
      role="img"
      aria-label="Players on the court from above"
    >
      <rect x={0} y={0} width={sideways ? 20 : 10} height={sideways ? 10 : 20} className="floor" />
      {COURT_LINES.map(([x1, y1, x2, y2], i) => (
        <line key={i} {...line(x1, y1, x2, y2)} className="court-line" />
      ))}
      <line {...line(-0.4, 10, 10.4, 10)} className="net" />
      {Object.entries(overlay.players).map(([pid, p]) => {
        const i = nearest(p.t, t, 0.5);
        if (i < 0) return null;
        const trail = [];
        for (let k = lowerBound(p.t, t - TRAIL_S); k <= i; k++) trail.push(P(p.xy[2 * k], p.xy[2 * k + 1]).join(","));
        const [x, y] = P(p.xy[2 * i], p.xy[2 * i + 1]);
        return (
          <g key={pid} style={{ color: PLAYER_COLORS[pid] }}>
            <polyline points={trail.join(" ")} className="trail" />
            <circle cx={x} cy={y} r={0.38} fill="currentColor" />
            <text x={x} y={y - 0.6} className="mm-label">
              {pid}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

function Timeline({
  rallies,
  best,
  total,
  t,
  onSeek,
}: {
  rallies: Rally[];
  best: Set<number>;
  total: number;
  t: number;
  onSeek: (t: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const pct = (x: number) => `${Math.min(100, Math.max(0, (x / total) * 100))}%`;
  return (
    <div
      className="timeline"
      ref={ref}
      onClick={(e) => {
        const r = ref.current?.getBoundingClientRect();
        if (r) onSeek(((e.clientX - r.left) / r.width) * total);
      }}
    >
      {rallies.map((r) => (
        <button
          key={r.index}
          className={best.has(r.index) ? "best" : ""}
          style={{
            left: pct(r.start_s),
            width: pct(Math.max(r.end_s - r.start_s, total * 0.003)),
            opacity: 0.35 + 0.65 * r.score,
          }}
          title={`Point ${r.index + 1} at ${formatTime(r.start_s)}: ${r.hits} shots, ${Math.round(r.end_s - r.start_s)} s`}
          aria-label={`Play point ${r.index + 1} at ${formatTime(r.start_s)}`}
          onClick={(e) => {
            e.stopPropagation();
            onSeek(r.start_s - 2);
          }}
        />
      ))}
      <div className="now" style={{ left: pct(t) }} />
    </div>
  );
}
