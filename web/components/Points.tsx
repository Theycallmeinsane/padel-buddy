"use client";

import { useMemo, useState } from "react";
import type { Api, Results } from "@/lib/api";
import { pointSpan, type PointEdits } from "@/lib/points";
import { endingLabel } from "@/lib/shots";
import { formatTime } from "./common";
import PointTrim from "./PointTrim";

/** Every point of the match, rally by rally: play it in the viewer, or download it on its own. */
export default function Points({
  api,
  matchId,
  data,
  names = {},
  onSeek,
  edits = {},
  durationS = null,
  onEdit,
  onReset,
}: {
  api: Api;
  matchId: string;
  data: Results;
  names?: Record<string, string>;
  onSeek?: (t: number) => void;
  edits?: PointEdits;
  durationS?: number | null;
  // Moving a point's start and end by hand; without these the points can't be adjusted.
  onEdit?: (index: number, start: number, end: number) => Promise<string | null>;
  onReset?: (index: number) => Promise<string | null>;
}) {
  const [adjusting, setAdjusting] = useState<number | null>(null);
  const rallies = useMemo(() => data.clips?.rallies ?? [], [data.clips]);
  const endings = useMemo(() => new Map((data.shots?.points ?? []).map((p) => [p.rally, p])), [data.shots]);
  if (!rallies.length) return null;
  const clips = rallies.filter((r) => r.clip).length;
  const span = (r: (typeof rallies)[number]) => pointSpan(r, data.clips, edits, durationS);
  const open = adjusting === null ? undefined : rallies.find((r) => r.index === adjusting);
  const openSpan = open ? span(open) : null;
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
        {clips && onEdit
          ? " Each download runs 5 s past where the point seemed to end; if a split is off, Adjust it and drag its start and end."
          : ""}
      </p>
      {open && openSpan && onEdit && onReset && (
        <PointTrim
          api={api}
          matchId={matchId}
          label={`point ${open.index + 1}`}
          span={[openSpan.start, openSpan.end]}
          edited={openSpan.edited}
          durationS={durationS}
          onSave={(a, b) => onEdit(open.index, a, b)}
          onReset={() => onReset(open.index)}
          onClose={() => setAdjusting(null)}
        />
      )}
      <ul className="points">
        {rallies.map((r) => {
          const ending = endings.get(r.index);
          const shots = ending?.shots ?? r.hits;
          const cut = r.clip ? span(r) : null;
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
                <span className={cut?.edited ? "edited" : "muted"} title={cut?.edited ? "Adjusted by hand" : undefined}>
                  {formatTime(cut?.edited ? cut.start : r.start_s)}
                  {cut?.edited ? " ✎" : ""}
                </span>
                <span className="muted">{Math.round(cut?.edited ? cut.end - cut.start : r.end_s - r.start_s)} s</span>
                <span className="muted">
                  {shots} shot{shots === 1 ? "" : "s"}
                </span>
                <span className="ending">{endingLabel(ending, names)}</span>
              </span>
              {r.clip ? (
                <span className="actions small">
                  {onEdit && (
                    <button className="link small" onClick={() => setAdjusting(r.index)}>
                      Adjust
                    </button>
                  )}
                  <a href={api.originalUrl(matchId, r.clip)} download>
                    Download
                  </a>
                </span>
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
