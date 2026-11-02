"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Api } from "@/lib/api";
import { moveEdge, trimWindow } from "@/lib/points";
import { formatTime } from "./common";
import { SkipButtons } from "./Video";

/**
 * Drag where a point starts and ends over the full match, watching the video
 * follow each handle, then save: the point's download is cut there.
 */
export default function PointTrim({
  api,
  matchId,
  label,
  span,
  edited,
  durationS,
  onSave,
  onReset,
  onClose,
}: {
  api: Api;
  matchId: string;
  label: string;
  span: [number, number];
  edited: boolean;
  durationS: number | null;
  onSave: (start: number, end: number) => Promise<string | null>;
  onReset: () => Promise<string | null>;
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [cut, setCut] = useState<[number, number]>(span);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const stopAt = useRef<number | null>(null);
  // The bar keeps the reach it opened with, so it doesn't slide under the handle being dragged.
  const win = useMemo(() => trimWindow(span[0], span[1], durationS), [span, durationS]);
  const [lo, hi] = win;
  const pct = (t: number) => `${((t - lo) / (hi - lo)) * 100}%`;

  useEffect(() => setCut(span), [span]);

  function seek(t: number) {
    const v = video.current;
    if (v) v.currentTime = t;
  }

  function move(edge: "start" | "end", t: number) {
    const next = moveEdge(cut, edge, t, win);
    setCut(next);
    stopAt.current = null;
    video.current?.pause();
    seek(edge === "start" ? next[0] : next[1]);
  }

  function here(edge: "start" | "end") {
    const t = video.current?.currentTime;
    if (t !== undefined) move(edge, t);
  }

  function preview() {
    const v = video.current;
    if (!v) return;
    v.currentTime = cut[0];
    stopAt.current = cut[1];
    v.play().catch(() => {});
  }

  async function run(action: () => Promise<string | null>) {
    setBusy(true);
    const problem = await action();
    setBusy(false);
    setError(problem);
    if (!problem) onClose();
  }

  const changed = cut[0] !== span[0] || cut[1] !== span[1];
  return (
    <div className="trim">
      <div className="row" style={{ marginBottom: 8 }}>
        <strong className="grow">Adjust {label}</strong>
        <button className="link small" onClick={onClose}>
          Close
        </button>
      </div>
      <video
        ref={video}
        className="reel"
        src={api.videoUrl(matchId)}
        preload="metadata"
        controls
        controlsList="nodownload"
        onLoadedMetadata={() => seek(cut[0])}
        onTimeUpdate={(e) => {
          if (stopAt.current !== null && e.currentTarget.currentTime >= stopAt.current) {
            e.currentTarget.pause();
            stopAt.current = null;
          }
        }}
      />
      <SkipButtons videoRef={video} />
      <div className="trim-bar" aria-label="Drag the handles to where the point starts and ends">
        <div className="trim-sel" style={{ left: pct(cut[0]), width: `calc(${pct(cut[1])} - ${pct(cut[0])})` }} />
        <input
          type="range"
          min={lo}
          max={hi}
          step={0.1}
          value={cut[0]}
          onChange={(e) => move("start", Number(e.target.value))}
          aria-label="Point starts at"
        />
        <input
          type="range"
          min={lo}
          max={hi}
          step={0.1}
          value={cut[1]}
          onChange={(e) => move("end", Number(e.target.value))}
          aria-label="Point ends at"
        />
      </div>
      <div className="row small muted" style={{ justifyContent: "space-between" }}>
        <span>{formatTime(lo)}</span>
        <span>
          <strong style={{ color: "var(--text)" }}>
            {formatTime(cut[0])} – {formatTime(cut[1])}
          </strong>{" "}
          · {Math.round(cut[1] - cut[0])} s
        </span>
        <span>{formatTime(hi)}</span>
      </div>
      <div className="row" style={{ marginTop: 8, gap: 8 }}>
        <button onClick={() => here("start")}>Start here</button>
        <button onClick={() => here("end")}>End here</button>
        <button onClick={preview}>▶ Preview</button>
        <span className="grow" />
        {edited && (
          <button className="link small" disabled={busy} onClick={() => run(onReset)}>
            Back to the automatic split
          </button>
        )}
        <button className="primary" disabled={busy || !changed} onClick={() => run(() => onSave(cut[0], cut[1]))}>
          Save
        </button>
      </div>
      {error && <p className="error small">{error}</p>}
    </div>
  );
}
