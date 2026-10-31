"use client";

import { useState } from "react";
import type { Fix, FixIn } from "@/lib/api";
import { PLAYERS, fixProblem, fixSummary, parseTime, playerName } from "@/lib/fixes";
import { formatTime } from "./common";

/** Saves a fix; resolves to an error message, or null when it was saved. */
export type SaveFix = (fix: FixIn) => Promise<string | null>;

/**
 * The fallback when the names on the boxes go wrong partway through and tags
 * don't put it right: from a moment on (or until another), the player shown
 * as one name is really another, and the two swap there.
 */
export function FixPanel({
  names,
  now,
  onSave,
  onClose,
}: {
  names: Record<string, string>;
  now: number; // match time where the video is
  onSave: SaveFix;
  onClose: () => void;
}) {
  const [start, setStart] = useState(formatTime(now));
  const [end, setEnd] = useState("");
  const [toEnd, setToEnd] = useState(false);
  const [player, setPlayer] = useState("");
  const [becomes, setBecomes] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const startS = parseTime(start);
  const endS = end.trim() ? parseTime(end) : null;
  const badEnd = end.trim() !== "" && endS === null;
  const problem = badEnd
    ? "Type the end as minutes:seconds, or leave it empty"
    : fixProblem(startS, endS, player, becomes);

  async function save() {
    if (problem || startS === null) return;
    setSaving(true);
    const failed = await onSave({ start_s: startS, end_s: endS, player, becomes, to_end: toEnd && endS === null });
    setSaving(false);
    if (failed) setError(failed);
    else onClose();
  }

  const options = (
    <>
      <option value="">choose…</option>
      {PLAYERS.map((pid) => (
        <option key={pid} value={pid}>
          {playerName(names, pid)}
          {names[pid] ? ` (${pid})` : ""}
        </option>
      ))}
    </>
  );

  return (
    <form
      className="fix-panel"
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <p className="small muted" style={{ marginTop: 0 }}>
        Names on the wrong players from some point on? Pause where it goes wrong and say who it really is. The
        two swap from there until the tracking next switches either of them, since it is often right again
        after that. Give an end time, or tick to the end of the match, to change that. Nothing else changes,
        and × takes it back.
      </p>
      <div className="row small">
        <label>
          From{" "}
          <input
            type="text"
            inputMode="numeric"
            size={7}
            value={start}
            onChange={(e) => setStart(e.target.value)}
            aria-label="Start, minutes:seconds"
          />
        </label>
        <button type="button" onClick={() => setStart(formatTime(now))}>
          Now
        </button>
        <label>
          to{" "}
          <input
            type="text"
            inputMode="numeric"
            size={7}
            placeholder={toEnd ? "the end" : "auto"}
            disabled={toEnd}
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            aria-label="End, minutes:seconds; empty to stop where the tracking switches them"
          />
        </label>
        <button type="button" onClick={() => setEnd(formatTime(now))} disabled={toEnd}>
          Now
        </button>
        <label className="toggle">
          <input
            type="checkbox"
            checked={toEnd}
            onChange={(e) => {
              setToEnd(e.target.checked);
              if (e.target.checked) setEnd("");
            }}
          />
          to the end of the match
        </label>
      </div>
      <div className="row small" style={{ marginTop: 8 }}>
        <label>
          the player shown as{" "}
          <select value={player} onChange={(e) => setPlayer(e.target.value)}>
            {options}
          </select>
        </label>
        <label>
          is really{" "}
          <select value={becomes} onChange={(e) => setBecomes(e.target.value)}>
            {options}
          </select>
        </label>
      </div>
      <div className="row small" style={{ marginTop: 8 }}>
        <button className="primary" type="submit" disabled={!!problem || saving}>
          {saving ? "Saving…" : "Swap them"}
        </button>
        <button type="button" onClick={onClose}>
          Cancel
        </button>
        {(error || (player && becomes && problem)) && <span className="error">{error || problem}</span>}
      </div>
    </form>
  );
}

/** Every fix as a chip: what it says (click to watch from there), and × to take it back. */
export function FixList({
  fixes,
  names,
  onSeek,
  onDelete,
}: {
  fixes: Fix[];
  names: Record<string, string>;
  onSeek: (t: number) => void;
  onDelete: (id: string) => void;
}) {
  if (!fixes.length) return null;
  return (
    <ul className="tags" aria-label="Players you put right">
      {fixes.map((f) => {
        const label = fixSummary(f, names, formatTime);
        return (
          <li key={f.id}>
            <button className="link" onClick={() => onSeek(f.start_s - 1)}>
              {label}
            </button>
            <button className="x" aria-label={`Take back: ${label}`} onClick={() => onDelete(f.id)}>
              ×
            </button>
          </li>
        );
      })}
    </ul>
  );
}
