"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Api, Check, CheckScore, Match } from "@/lib/api";
import { CHECK_EVERY_S, checkFrames, checksAt, defaultCheckStart, framesDone, percent } from "@/lib/checks";
import { formatTime } from "./common";

/**
 * Checking the tracking by hand: on a frame every 3 s over 10 minutes, click
 * each player and say who they really are. Checks never change who is who;
 * they score it, so a change to the tracking is kept only if the score goes up.
 */
export default function CheckTracking({
  api,
  match,
  names,
  onMatch,
}: {
  api: Api;
  match: Match;
  names: Record<string, string>;
  onMatch: (m: Match) => void;
}) {
  const checks = useMemo(() => match.checks ?? [], [match.checks]);
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState(() =>
    checks.length ? Math.min(...checks.map((c) => c.time_s)) : defaultCheckStart(match.duration_s),
  );
  const frames = useMemo(() => checkFrames(start, match.duration_s), [start, match.duration_s]);
  const [i, setI] = useState(() => {
    // Carry on from the first frame not yet done.
    const k = frames.findIndex((t) => checksAt(checks, t).length < 4);
    return k < 0 ? 0 : k;
  });
  const t = frames[Math.min(i, frames.length - 1)] ?? 0;
  const here = checksAt(checks, t);
  const [size, setSize] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<[number, number] | null>(null);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [score, setScore] = useState<CheckScore | null>(null);
  const img = useRef<HTMLImageElement>(null);
  const people = useMemo(() => {
    const seen = new Map<string, string>();
    for (const n of [...Object.values(names), ...checks.map((c) => c.name)]) seen.set(n.toLowerCase(), n);
    return [...seen.values()];
  }, [names, checks]);

  const loadScore = useCallback(async () => {
    if (!checks.length) return setScore(null);
    try {
      setScore(await api.checkScore(match.id));
    } catch {
      setScore(null);
    }
  }, [api, match.id, checks.length]);

  useEffect(() => {
    if (!open) loadScore();
  }, [open, loadScore, match.job?.finished_at]);

  function go(k: number) {
    setDraft(null);
    setError(null);
    setI(Math.max(0, Math.min(frames.length - 1, k)));
  }

  function click(e: React.MouseEvent<HTMLImageElement>) {
    if (!img.current) return;
    const r = img.current.getBoundingClientRect();
    setTyped("");
    setDraft([
      ((e.clientX - r.left) * img.current.naturalWidth) / r.width,
      ((e.clientY - r.top) * img.current.naturalHeight) / r.height,
    ]);
  }

  async function save(name: string) {
    if (!draft || !name.trim()) return;
    try {
      const m = await api.addCheck(match.id, { time_s: t, x: draft[0], y: draft[1], name: name.trim() });
      onMatch(m);
      setDraft(null);
      setError(null);
      // All four checked here: straight on to the next frame.
      if (checksAt(m.checks ?? [], t).length >= 4 && i < frames.length - 1) go(i + 1);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(c: Check) {
    try {
      onMatch(await api.deleteCheck(match.id, c.id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const done = framesDone(checks, frames);

  return (
    <section className="card">
      <h2>Check the tracking</h2>
      <p className="small muted" style={{ marginTop: 0 }}>
        Say who each player really is on a frame every {CHECK_EVERY_S} s over 10 minutes. This never changes who is
        who: it scores the tracking, so each improvement can be measured. About an hour for all {frames.length}{" "}
        frames; even 50 give a first score.
      </p>
      {score && <ScoreCard score={score} />}
      {!open ? (
        <button className="primary" onClick={() => setOpen(true)}>
          {checks.length ? `Carry on checking (${done} of ${frames.length} frames done)` : "Start checking"}
        </button>
      ) : (
        <>
          <div className="frame">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={img}
              src={api.frameUrl(match.id, t)}
              alt="Video frame to check the players on"
              onClick={click}
              onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
            />
            {size &&
              here.map((c) => (
                <div
                  key={c.id}
                  className="pt named"
                  style={{ left: `${(c.x / size[0]) * 100}%`, top: `${(c.y / size[1]) * 100}%` }}
                >
                  <span>{c.name}</span>
                </div>
              ))}
            {size && draft && (
              <form
                className="name-box"
                style={{
                  left: `${Math.min(Math.max((draft[0] / size[0]) * 100, 18), 82)}%`,
                  top: `${(draft[1] / size[1]) * 100}%`,
                  transform:
                    draft[1] / size[1] > 0.55 ? "translate(-50%, calc(-100% - 10px))" : "translate(-50%, 10px)",
                }}
                onSubmit={(e) => {
                  e.preventDefault();
                  save(typed);
                }}
                onKeyDown={(e) => e.key === "Escape" && setDraft(null)}
              >
                <div className="row small" style={{ flexWrap: "wrap" }}>
                  {people.map((n) => (
                    <button key={n} type="button" onClick={() => save(n)}>
                      {n}
                    </button>
                  ))}
                </div>
                <input
                  type="text"
                  autoFocus={!people.length}
                  placeholder={people.length ? "Someone else" : "Who is this?"}
                  aria-label="Player's name"
                  maxLength={60}
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                />
                <div className="row small">
                  <button className="primary" type="submit" disabled={!typed.trim()}>
                    Save
                  </button>
                  <button type="button" onClick={() => setDraft(null)}>
                    Cancel
                  </button>
                </div>
              </form>
            )}
          </div>
          {error && <p className="error small">{error}</p>}
          {here.length > 0 && (
            <ul className="tags" aria-label="Checked on this frame">
              {here.map((c) => (
                <li key={c.id}>
                  <span>{c.name}</span>
                  <button className="x" aria-label={`Take back ${c.name}`} onClick={() => remove(c)}>
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          <div className="row small" style={{ marginTop: 8 }}>
            <span className="muted grow">
              Frame {i + 1} of {frames.length} at {formatTime(t)} · {here.length} of 4 checked · {done} frames done
            </span>
            <button onClick={() => go(i - 1)} disabled={i === 0}>
              Previous
            </button>
            <button className={here.length >= 4 ? "primary" : ""} onClick={() => go(i + 1)} disabled={i >= frames.length - 1}>
              Next frame
            </button>
          </div>
          <p className="small muted">
            Click every player you can see, and skip anyone hidden. Clicking the same spot again replaces a wrong name.
          </p>
          {!checks.length && match.duration_s ? (
            <div className="row small">
              <span className="muted">Start checking at {formatTime(start)}</span>
              <input
                className="grow"
                type="range"
                min={0}
                max={Math.max(0, Math.floor(match.duration_s - 60))}
                step={CHECK_EVERY_S}
                value={start}
                onChange={(e) => {
                  setStart(Number(e.target.value));
                  setI(0);
                }}
                aria-label="Where in the match to start checking"
              />
            </div>
          ) : null}
          <div className="row" style={{ marginTop: 12 }}>
            <button className="primary" onClick={() => setOpen(false)}>
              Done for now, show the score
            </button>
          </div>
        </>
      )}
    </section>
  );
}

/** The tracking's score against the checks: overall, by side of the net, and per person. */
function ScoreCard({ score }: { score: CheckScore }) {
  return (
    <div className="small" style={{ marginBottom: 12 }}>
      <p style={{ margin: "0 0 6px" }}>
        <strong>Named right {percent(score.right)}</strong> of {score.checks} checks · wrong {percent(score.wrong)} ·
        not found {percent(score.missed)} · {score.switches} switch{score.switches === 1 ? "" : "es"}
        {score.switches_per_10_min !== null ? ` (${score.switches_per_10_min} per 10 min)` : ""}
      </p>
      <p className="muted" style={{ margin: "0 0 6px" }}>
        Near side right {percent(score.sides.near.right)} · far side right {percent(score.sides.far.right)} (of the
        players found)
      </p>
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th>Checks</th>
            <th>Right</th>
            <th>Wrong</th>
            <th>Not found</th>
            <th>Switches</th>
          </tr>
        </thead>
        <tbody>
          {score.people.map((p) => (
            <tr key={p.name}>
              <td>{p.name}</td>
              <td>{p.checks}</td>
              <td>{percent(p.right)}</td>
              <td>{percent(p.wrong)}</td>
              <td>{percent(p.missed)}</td>
              <td>{p.switches}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {Object.keys(score.assumed).length > 0 && (
        <p className="muted" style={{ margin: "6px 0 0" }}>
          Not named in this match, so scored by where the checks landed:{" "}
          {Object.entries(score.assumed)
            .map(([pid, n]) => `${n} as ${pid}`)
            .join(", ")}
          .
        </p>
      )}
    </div>
  );
}
