"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { Api, BallTrack, Bounce, Overlay, PointEnding, Rally, Results, Shot, Tag } from "@/lib/api";
import { KIND_LABELS } from "@/lib/shots";
import { playerAt, toSource } from "@/lib/tags";
import { lowerBound, nearest, toMatchTime, toVideoTime, type Segment } from "@/lib/timeline";
import { formatTime } from "./common";
import { NameBox, TagList, type SaveTag } from "./Tags";

// Same colours as the engine's preview video: team A in blues, team B warm.
export const PLAYER_COLORS: Record<string, string> = { P1: "#1e8ceb", P2: "#28dce6", P3: "#f58c28", P4: "#e63c3c" };
const TEAM_COLORS: Record<string, string> = { A: "#2d6fd6", B: "#e0702b" };
const SHOT_LABEL_S = 1.0; // a shot's name shows over the hitter this long
const BOX_GAP_S = 0.3; // hide a player when their nearest sample is further away than this
const TRAIL_S = 2;
const BALL_COLOR = "#f4ff3d";
const BALL_TRAIL_S = 0.4;
const BOUNCE_SHOW_S = 1.2; // a bounce ring on the video fades over this long
const MAP_BOUNCES_S = 6; // bounces stay on the minimap this long

// Painted lines in court metres (x across, y from the near wall), net at y = 10.
const COURT_LINES: [number, number, number, number][] = [
  [0, 3.05, 10, 3.05],
  [0, 16.95, 10, 16.95],
  [5, 2.85, 5, 17.15],
];

type Layers = { lines: boolean; players: boolean; ball: boolean; shots: boolean; minimap: boolean };
type Marks = { shots: Shot[]; names: Record<string, string>; focus: string | null };
type Source = "points" | "full";
type Draft = { t: number; x: number; y: number; player: string; left: number; top: number };

export default function MatchViewer({
  api,
  matchId,
  durationS,
  data,
  names = {},
  focus = null,
  seekTo = null,
  tags = [],
  suggestions = [],
  onTag,
  onDeleteTag,
  updating = false,
  live = false,
  version = "",
}: {
  api: Api;
  matchId: string;
  durationS: number | null;
  data: Results;
  names?: Record<string, string>;
  focus?: string | null; // follow one player: the others are dimmed
  seekTo?: { t: number; n: number } | null; // a request from outside to play from match time t
  tags?: Tag[];
  suggestions?: string[]; // names to offer when tagging
  onTag?: SaveTag;
  onDeleteTag?: (id: string) => void;
  updating?: boolean; // who is who is being decided again after a tag
  live?: boolean; // the results so far, while the match is analysed
  version?: string; // changes when the outputs were rewritten in place
}) {
  const clips = data.clips;
  const rallies = useMemo(() => clips?.rallies ?? [], [clips]);
  const overlayFile = data.tracking?.outputs.overlay ?? null;
  const ballFile = data.ball?.outputs.track ?? null;
  const condensed = clips?.condensed && clips.condensed_segments?.length ? clips.condensed : null;
  const total = durationS || Math.max(1, ...rallies.map((r) => r.end_s));
  const best = useMemo(() => new Set(clips?.clips.flatMap((c) => c.rallies ?? []) ?? []), [clips]);
  const shots = useMemo(() => data.shots?.shots ?? [], [data.shots]);
  const endings = useMemo(() => new Map((data.shots?.points ?? []).map((p) => [p.rally, p])), [data.shots]);
  const marks = useMemo<Marks>(() => ({ shots, names, focus }), [shots, names, focus]);

  const [source, setSource] = useState<Source>(condensed ? "points" : "full");
  const [fullFailed, setFullFailed] = useState(false);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [ball, setBall] = useState<BallTrack | null>(null);
  const [layers, setLayers] = useState<Layers>({ lines: true, players: true, ball: true, shots: true, minimap: true });
  const [matchTime, setMatchTime] = useState(0);
  const [tagging, setTagging] = useState(false);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [hint, setHint] = useState<string | null>(null);
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
  }, [api, matchId, overlayFile, version]);

  useEffect(() => {
    if (!ballFile) return;
    let live = true;
    api.ballTrack(matchId, ballFile).then(
      (b) => live && setBall(b),
      (e) => live && setOverlayError((e as Error).message),
    );
    return () => {
      live = false;
    };
  }, [api, matchId, ballFile, version]);

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
      paint(canvas, video, overlay, ball, layers, t, marks);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [overlay, ball, layers, segments, marks]);

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

  useEffect(() => {
    if (!seekTo) return;
    seek(seekTo.t);
    videoRef.current?.closest("section")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [seekTo, seek]);

  function startTagging(on: boolean) {
    setTagging(on);
    setDraft(null);
    setHint(on ? "Pause where you can see the player, then click their box." : null);
    if (on) videoRef.current?.pause();
  }

  function pick(e: React.MouseEvent<HTMLCanvasElement>) {
    const video = videoRef.current;
    if (!tagging || !overlay || !video?.videoWidth) return;
    video.pause();
    const r = e.currentTarget.getBoundingClientRect();
    const at = toSource(
      e.clientX - r.left, e.clientY - r.top, r.width, r.height, video.videoWidth, video.videoHeight, overlay.size,
    );
    const t = toMatchTime(segments, video.currentTime);
    const player = at && playerAt(overlay, t, at[0], at[1], BOX_GAP_S);
    if (!at || !player) {
      setDraft(null);
      setHint("Nobody tracked there at this moment. Click inside a player's box.");
      return;
    }
    setHint(null);
    setDraft({
      t, x: at[0], y: at[1], player,
      left: ((e.clientX - r.left) / r.width) * 100, top: ((e.clientY - r.top) / r.height) * 100,
    });
  }

  function switchTo(next: Source) {
    resumeAt.current = matchTime;
    setSource(next);
  }

  const point = rallies.find((r) => r.start_s - 2 <= matchTime && matchTime <= r.end_s + 2);
  const ending = point ? endings.get(point.index) : undefined;
  const src = source === "points" && condensed ? api.fileUrl(matchId, condensed) : api.videoUrl(matchId);

  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 12 }}>
        <h2 className="grow" style={{ margin: 0 }}>
          Watch
        </h2>
        {overlay && onTag && (
          <button className={tagging ? "primary" : ""} onClick={() => startTagging(!tagging)}>
            {tagging ? "Done naming" : "Name a player"}
          </button>
        )}
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
            controlsList="nodownload"
            playsInline
            preload="metadata"
            onLoadedMetadata={(e) => {
              // A video track the browser can't decode (HEVC in Chrome or Edge on many PCs) raises
              // no error: the sound plays over a blank picture and the video reports no size.
              if (source === "full" && e.currentTarget.videoWidth === 0) {
                setFullFailed(true);
                if (condensed) setSource("points");
                return;
              }
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
          <canvas ref={canvasRef} aria-hidden="true" className={tagging ? "tagging" : undefined} onClick={pick} />
          {draft && onTag && (
            <NameBox
              left={draft.left}
              top={draft.top}
              suggestions={suggestions}
              onSave={async (name, notPlayer) => {
                const error = await onTag({
                  time_s: draft.t, x: draft.x, y: draft.y, name, player: draft.player, not_player: notPlayer,
                });
                if (!error) setDraft(null);
                return error;
              }}
              onCancel={() => setDraft(null)}
            />
          )}
        </div>
        {overlay && layers.minimap && (
          <Minimap
            overlay={overlay}
            bounces={layers.ball ? ball?.bounces : undefined}
            t={matchTime}
            sideways={narrow}
            names={names}
          />
        )}
      </div>

      <div className="row small" style={{ marginTop: 8 }}>
        <span className="grow muted">
          {formatTime(matchTime)}
          {point
            ? ` · point ${point.index + 1} of ${rallies.length} · ${point.hits} shots${best.has(point.index) ? " · one of the best" : ""}${ending ? endingText(ending, names) : ""}`
            : ""}
        </span>
        {overlay &&
          (
            [
              ["lines", "Court lines"],
              ["players", "Players"],
              ...(ball ? ([["ball", "Ball"]] as const) : []),
              ...(shots.length ? ([["shots", "Shots"]] as const) : []),
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

      {rallies.length > 0 && (
        <Timeline rallies={rallies} best={best} endings={endings} total={total} t={matchTime} onSeek={seek} />
      )}

      <p className="small" style={{ margin: "8px 0 0" }}>
        <a href={api.originalVideoUrl(matchId)} download>
          Download the full match (original file)
        </a>
      </p>
      {tagging && hint && <p className="small muted">{hint}</p>}
      {updating && <p className="small muted">Updating who is who from your tags. The names here change when it&apos;s done.</p>}
      {onDeleteTag && <TagList tags={tags} onSeek={seek} onDelete={onDeleteTag} />}

      {fullFailed && (
        <p className="muted small">
          This browser can&apos;t show the original video (phone recordings are often HEVC). Padel Buddy is
          making a copy that plays everywhere; reload this page in a few minutes to watch the full match.{" "}
          {condensed
            ? "Points only plays everywhere."
            : live
              ? "Points only plays everywhere once the analysis finishes; meanwhile each point's clip is under Every point as soon as it is cut."
              : "Analyse again to get a condensed match that plays everywhere."}
        </p>
      )}
      {!overlayFile && (
        <p className="muted small">
          {live
            ? "Court lines and the players appear here once the first few minutes are tracked."
            : "Calibrate the court and analyse again to see court lines and the players."}
        </p>
      )}
      {data.ball && (
        <p className="muted small">
          Ball found in {Math.round(data.ball.seen_fraction * 100)}% of the frames during points, with{" "}
          {data.ball.bounces} bounces on the court.
        </p>
      )}
      {overlayError && <p className="error small">Couldn&apos;t load the tracking overlay: {overlayError}</p>}
    </section>
  );
}

function endingText(p: PointEnding, names: Record<string, string>): string {
  if (p.ending === "unknown" || !p.won_by) return "";
  const who = p.last_player ? names[p.last_player] ?? p.last_player : `team ${p.last_team}`;
  if (p.ending === "winner") return ` · won by team ${p.won_by}: winner by ${who}`;
  return ` · won by team ${p.won_by}: ${who} hit ${p.detail === "net" ? "the net" : "it out"}`;
}

function paint(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  overlay: Overlay | null,
  ball: BallTrack | null,
  layers: Layers,
  t: number,
  marks: Marks,
) {
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
  const size = overlay?.size ?? ball?.size;
  if (!size || !video.videoWidth || !(layers.lines || layers.players || layers.ball)) return;

  // The picture is letterboxed inside the element, and may be smaller than the
  // source the engine measured in (the condensed match is 720p).
  const fit = Math.min(w / video.videoWidth, h / video.videoHeight);
  const pw = video.videoWidth * fit;
  const ph = video.videoHeight * fit;
  const s = pw / size[0];
  ctx.setTransform(dpr * s, 0, 0, dpr * s, (dpr * (w - pw)) / 2, (dpr * (h - ph)) / 2);
  const px = 1 / s; // one screen pixel, in source pixels

  if (overlay && layers.lines) {
    ctx.lineWidth = 1.5 * px;
    for (const line of overlay.lines) {
      ctx.strokeStyle = line.name === "net" ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.65)";
      ctx.beginPath();
      line.points.forEach(([u, v], i) => (i ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
      ctx.stroke();
    }
  }
  if (overlay && layers.players) {
    ctx.lineWidth = 2 * px;
    ctx.font = `600 ${13 * px}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    // The shot each player just played, named over their box for a moment.
    const played = new Map<string, Shot>();
    if (layers.shots) {
      for (let k = lowerBoundBy(marks.shots, t - SHOT_LABEL_S); k < marks.shots.length && marks.shots[k].time_s <= t; k++) {
        const shot = marks.shots[k];
        if (shot.player && shot.kind !== "unknown" && KIND_LABELS[shot.kind]) played.set(shot.player, shot);
      }
    }
    for (const [pid, p] of Object.entries(overlay.players)) {
      const i = nearest(p.t, t, BOX_GAP_S);
      if (i < 0) continue;
      const [x1, y1, x2, y2] = p.box.slice(4 * i, 4 * i + 4);
      const focused = marks.focus === pid;
      ctx.globalAlpha = marks.focus && !focused ? 0.35 : 1;
      ctx.lineWidth = (focused ? 3.5 : 2) * px;
      ctx.strokeStyle = ctx.fillStyle = PLAYER_COLORS[pid] ?? "#ffffff";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const shot = played.get(pid);
      ctx.fillText(`${marks.names[pid] ?? pid}${shot ? ` · ${KIND_LABELS[shot.kind]}` : ""}`, x1, y1 - 3 * px);
    }
    ctx.globalAlpha = 1;
  }
  if (ball && layers.ball) paintBall(ctx, ball, t, px);
}

function paintBall(ctx: CanvasRenderingContext2D, ball: BallTrack, t: number, px: number) {
  const i = nearest(ball.t, t, 1.5 / ball.fps);
  if (i >= 0) {
    // A short tail from the same flight, then the ball itself.
    ctx.strokeStyle = BALL_COLOR;
    ctx.lineWidth = 2 * px;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    let started = false;
    for (let k = lowerBound(ball.t, t - BALL_TRAIL_S); k <= i; k++) {
      if (ball.flight[k] !== ball.flight[i]) continue;
      if (started) ctx.lineTo(ball.uv[2 * k], ball.uv[2 * k + 1]);
      else ctx.moveTo(ball.uv[2 * k], ball.uv[2 * k + 1]);
      started = true;
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(ball.uv[2 * i], ball.uv[2 * i + 1], 6 * px, 0, 2 * Math.PI);
    ctx.stroke();
  }
  // Rings that grow and fade where the ball just bounced.
  ctx.lineWidth = 2 * px;
  for (const b of recentBounces(ball.bounces, t, BOUNCE_SHOW_S)) {
    const age = (t - b.time_s) / BOUNCE_SHOW_S;
    ctx.globalAlpha = 1 - age;
    ctx.strokeStyle = BALL_COLOR;
    ctx.beginPath();
    ctx.ellipse(b.u, b.v, (8 + 22 * age) * px, (4 + 9 * age) * px, 0, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** First shot at or after `t` (shots are sorted by time). */
function lowerBoundBy(shots: Shot[], t: number): number {
  let lo = 0;
  let hi = shots.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (shots[mid].time_s < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Bounces from the last `within` seconds up to `t` (bounces are sorted by time). */
function recentBounces(bounces: Bounce[], t: number, within: number): Bounce[] {
  const out = [];
  for (let k = bounces.length - 1; k >= 0 && bounces[k].time_s > t - within; k--) {
    if (bounces[k].time_s <= t) out.push(bounces[k]);
  }
  return out;
}

function Minimap({
  overlay,
  bounces,
  t,
  sideways,
  names,
}: {
  overlay: Overlay;
  bounces?: Bounce[];
  t: number;
  sideways: boolean;
  names: Record<string, string>;
}) {
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
      {bounces &&
        recentBounces(bounces, t, MAP_BOUNCES_S).map((b) => {
          const [x, y] = P(b.court_x, b.court_y);
          return (
            <circle
              key={b.time_s}
              cx={x}
              cy={y}
              r={0.22}
              className="bounce"
              style={{ opacity: 1 - (t - b.time_s) / MAP_BOUNCES_S }}
            />
          );
        })}
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
              {names[pid]?.split(" ")[0].slice(0, 8) ?? pid}
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
  endings,
  total,
  t,
  onSeek,
}: {
  rallies: Rally[];
  best: Set<number>;
  endings: Map<number, PointEnding>;
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
            borderBottom: endings.get(r.index)?.won_by ? `3px solid ${TEAM_COLORS[endings.get(r.index)!.won_by!]}` : undefined,
          }}
          title={`Point ${r.index + 1} at ${formatTime(r.start_s)}: ${r.hits} shots, ${Math.round(r.end_s - r.start_s)} s${
            endings.get(r.index)?.won_by ? `, won by team ${endings.get(r.index)!.won_by}` : ""
          }`}
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
