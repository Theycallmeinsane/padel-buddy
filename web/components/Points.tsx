"use client";

import { useMemo } from "react";
import type { Api, Results } from "@/lib/api";
import { endingLabel } from "@/lib/shots";
import { formatTime } from "./common";

/** Every point of the match, rally by rally: play it in the viewer, or download it on its own. */
export default function Points({
  api,
  matchId,
  data,
  names = {},
  onSeek,
}: {
  api: Api;
  matchId: string;
  data: Results;
  names?: Record<string, string>;
  onSeek?: (t: number) => void;
}) {
  const rallies = useMemo(() => data.clips?.rallies ?? [], [data.clips]);
  const endings = useMemo(() => new Map((data.shots?.points ?? []).map((p) => [p.rally, p])), [data.shots]);
  if (!rallies.length) return null;
  const clips = rallies.filter((r) => r.clip).length;
  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 4 }}>
        <h2 className="grow" style={{ margin: 0 }}>
          Every point
        </h2>
        <span className="muted small">{rallies.length} points</span>
      </div>
      <p className="muted small" style={{ marginTop: 0 }}>
        {onSeek ? "Play any point in the viewer above" : "Every point of the match"}
        {clips ? ", or download it as a clip of its own." : "."}
      </p>
      <ul className="points">
        {rallies.map((r) => {
          const ending = endings.get(r.index);
          const shots = ending?.shots ?? r.hits;
          return (
            <li key={r.index}>
              {onSeek ? (
                <button className="play" onClick={() => onSeek(r.start_s - 2)} aria-label={`Play point ${r.index + 1}`}>
                  ▶
                </button>
              ) : (
                <span />
              )}
              <span className="info">
                <strong>Point {r.index + 1}</strong>
                <span className="muted">{formatTime(r.start_s)}</span>
                <span className="muted">{Math.round(r.end_s - r.start_s)} s</span>
                <span className="muted">
                  {shots} shot{shots === 1 ? "" : "s"}
                </span>
                <span className="ending">{endingLabel(ending, names)}</span>
              </span>
              {r.clip ? (
                <a className="small" href={api.fileUrl(matchId, r.clip)} download={r.clip.split("/").pop()}>
                  Download
                </a>
              ) : (
                <span />
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
