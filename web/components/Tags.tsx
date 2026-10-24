"use client";

import { useMemo, useRef, useState } from "react";
import type { Api, Tag, TagIn } from "@/lib/api";
import { formatTime } from "./common";

/** Saves a tag; resolves to an error message, or null when it was saved. */
export type SaveTag = (tag: TagIn) => Promise<string | null>;

/** "Who is this?", shown over the picture where someone was clicked (left/top in % of the picture). */
export function NameBox({
  left,
  top,
  suggestions,
  onSave,
  onCancel,
}: {
  left: number;
  top: number;
  suggestions: string[];
  onSave: (name: string) => Promise<string | null>;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const above = top > 55; // keep the box on the picture
  return (
    <form
      className="name-box"
      style={{
        left: `${Math.min(Math.max(left, 18), 82)}%`,
        top: `${top}%`,
        transform: above ? "translate(-50%, calc(-100% - 10px))" : "translate(-50%, 10px)",
      }}
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        setError(await onSave(name.trim()));
        setSaving(false);
      }}
      onKeyDown={(e) => e.key === "Escape" && onCancel()}
    >
      <input
        type="text"
        autoFocus
        list="known-names"
        placeholder="Who is this?"
        aria-label="Player's name"
        maxLength={60}
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <datalist id="known-names">
        {suggestions.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
      <div className="row small">
        <button className="primary" type="submit" disabled={!name.trim() || saving}>
          {saving ? "Saving…" : "Save"}
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
      {error && <p className="error small" style={{ margin: 0 }}>{error}</p>}
    </form>
  );
}

/** Every tag as a chip: who, when (click to watch that moment), and × to take it back. */
export function TagList({
  tags,
  onSeek,
  onDelete,
}: {
  tags: Tag[];
  onSeek?: (t: number) => void;
  onDelete: (id: string) => void;
}) {
  if (!tags.length) return null;
  return (
    <ul className="tags" aria-label="Players you named">
      {[...tags]
        .sort((a, b) => a.time_s - b.time_s)
        .map((t) => {
          const label = `${t.name} · ${formatTime(t.time_s)}`;
          return (
            <li
              key={t.id}
              className={t.applied === false ? "missed" : ""}
              title={
                t.applied === false
                  ? "Not applied: no tracked player there, or it contradicts another tag. Tag them at another moment."
                  : t.applied === null
                    ? "Applies when the match is analysed"
                    : undefined
              }
            >
              {onSeek ? (
                <button className="link" onClick={() => onSeek(t.time_s - 1)}>
                  {label}
                </button>
              ) : (
                <span>{label}</span>
              )}
              {t.applied === false && <span className="muted">· not applied</span>}
              <button className="x" aria-label={`Take back ${label}`} onClick={() => onDelete(t.id)}>
                ×
              </button>
            </li>
          );
        })}
    </ul>
  );
}

/** Before analysis: click each player on a frame and name them. */
export function PlayerTagger({
  api,
  matchId,
  durationS,
  tags,
  suggestions,
  onTag,
  onDelete,
  onDone,
}: {
  api: Api;
  matchId: string;
  durationS: number | null;
  tags: Tag[];
  suggestions: string[];
  onTag: SaveTag;
  onDelete: (id: string) => void;
  onDone: () => void;
}) {
  const [t, setT] = useState(Math.min(5, durationS ?? 5));
  const [size, setSize] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<[number, number] | null>(null);
  const img = useRef<HTMLImageElement>(null);
  const frameUrl = useMemo(() => api.frameUrl(matchId, t), [api, matchId, t]);
  const here = tags.filter((tag) => Math.abs(tag.time_s - t) < 0.5);

  function click(e: React.MouseEvent<HTMLImageElement>) {
    if (!img.current) return;
    const r = img.current.getBoundingClientRect();
    setDraft([
      ((e.clientX - r.left) * img.current.naturalWidth) / r.width,
      ((e.clientY - r.top) * img.current.naturalHeight) / r.height,
    ]);
  }

  return (
    <div>
      <p className="small muted" style={{ marginTop: 0 }}>
        Click each player and type their name, on a frame where you can see them clearly. The names follow them
        through the whole match. You can also name or correct anyone later, while watching.
      </p>
      <div className="frame">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={img}
          src={frameUrl}
          alt="Video frame to name the players on"
          onClick={click}
          onLoad={(e) => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
        />
        {size &&
          here.map((tag) => (
            <div
              key={tag.id}
              className="pt named"
              style={{ left: `${(tag.x / size[0]) * 100}%`, top: `${(tag.y / size[1]) * 100}%` }}
            >
              <span>{tag.name}</span>
            </div>
          ))}
        {size && draft && (
          <NameBox
            left={(draft[0] / size[0]) * 100}
            top={(draft[1] / size[1]) * 100}
            suggestions={suggestions}
            onSave={async (name) => {
              const error = await onTag({ time_s: t, x: draft[0], y: draft[1], name, player: null });
              if (!error) setDraft(null);
              return error;
            }}
            onCancel={() => setDraft(null)}
          />
        )}
      </div>
      {durationS ? (
        <div className="row small" style={{ marginTop: 8 }}>
          <span className="muted">Frame at {formatTime(t)}</span>
          <input
            className="grow"
            type="range"
            min={0}
            max={Math.max(0, Math.floor(durationS - 1))}
            value={t}
            onChange={(e) => {
              setDraft(null);
              setT(Number(e.target.value));
            }}
            aria-label="Choose a frame where the players are easy to see"
          />
        </div>
      ) : null}
      <TagList tags={tags} onDelete={onDelete} />
      <div className="row" style={{ marginTop: 12 }}>
        <button className="primary" onClick={onDone}>
          Done
        </button>
      </div>
    </div>
  );
}
