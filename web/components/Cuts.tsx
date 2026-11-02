"use client";

import { useState } from "react";
import type { Api, Cut, Match } from "@/lib/api";
import { parseCuts } from "@/lib/cuts";
import { formatTime } from "./common";
import { Video } from "./Video";

/** Type start and end times, and each becomes its own clip of the full match. */
export default function Cuts({
  api,
  match,
  onMatch,
}: {
  api: Api;
  match: Match;
  onMatch: (m: Match) => void;
}) {
  const [text, setText] = useState("");
  const [problems, setProblems] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const cuts = match.cuts ?? [];
  const waiting = cuts.filter((c) => c.status === "waiting").length;

  async function submit() {
    const { ranges, errors } = parseCuts(text, match.duration_s);
    setProblems(errors.map((e) => `"${e.text}": ${e.message}`));
    if (errors.length) return;
    if (!ranges.length) {
      setProblems(["Type at least one clip, e.g. 1:05 - 1:42"]);
      return;
    }
    setSaving(true);
    try {
      onMatch(await api.addCuts(match.id, ranges));
      setText("");
    } catch (e) {
      setProblems([(e as Error).message]);
    } finally {
      setSaving(false);
    }
  }

  async function remove(c: Cut) {
    if (!confirm(`Delete the clip ${label(c)}?`)) return;
    try {
      onMatch(await api.deleteCut(match.id, c.id));
    } catch (e) {
      setProblems([(e as Error).message]);
    }
  }

  return (
    <section className="card">
      <h2>Cut clips from times</h2>
      <p className="muted small" style={{ marginTop: 0 }}>
        One clip per line: start - end, then a name if you like. Times are minutes:seconds (1:05) or
        hours:minutes:seconds (1:02:03). Each clip runs 5 seconds past the end you type, so the point
        isn't cut short.
      </p>
      <textarea
        className="cuts-input"
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={"1:05 - 1:42 Great lob\n12:30 - 13:10\n1:02:03 - 1:02:40 Smash"}
        aria-label="Clips to cut, one per line"
      />
      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" onClick={submit} disabled={saving || !text.trim()}>
          {saving ? "Adding…" : "Cut clips"}
        </button>
        {waiting > 0 && (
          <span className="small muted">
            Cutting {waiting} clip{waiting === 1 ? "" : "s"}… they wait while an analysis is running.
          </span>
        )}
      </div>
      {problems.map((p) => (
        <p key={p} className="error small" style={{ margin: "6px 0 0" }}>
          {p}
        </p>
      ))}
      {cuts.length > 0 && (
        <div className="grid" style={{ marginTop: 12 }}>
          {cuts.map((c) => (
            <div className="clip cut" key={c.id}>
              {c.status === "done" ? (
                <Video controls controlsList="nodownload" preload="metadata" src={api.cutVideoUrl(match.id, c.id)} />
              ) : (
                <div className="cut-wait small">
                  {c.status === "waiting" ? "Cutting…" : <span className="error">Couldn&apos;t cut it: {c.error}</span>}
                </div>
              )}
              <div className="meta">
                <span>{label(c)}</span>
                <span className="muted">{formatTime(c.end_s - c.start_s)}</span>
              </div>
              <div className="meta">
                {c.status === "done" ? (
                  <a className="small" href={api.cutOriginalUrl(match.id, c.id)} download>
                    Download in original quality
                  </a>
                ) : (
                  <span />
                )}
                <button className="link small" onClick={() => remove(c)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function label(c: Cut): string {
  const span = `${formatTime(c.start_s)} – ${formatTime(c.end_s)}`;
  return c.name ? `${c.name} (${span})` : span;
}
