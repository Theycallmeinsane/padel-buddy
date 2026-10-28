"use client";

import { useMemo, useRef, useState } from "react";
import type { Api, Tag, TagIn } from "@/lib/api";
import { LABEL_GOAL, labelCounts, labelGap, labelMoments } from "@/lib/tags";
import { formatTime } from "./common";

/** Saves a tag; resolves to an error message, or null when it was saved. */
export type SaveTag = (tag: TagIn) => Promise<string | null>;

/** What a tag says: the name, or that the person is not a player. */
export function tagLabel(tag: Tag): string {
  return tag.not_player ? "Not a player" : tag.name;
}

/**
 * "Who is this?", shown over the picture where someone was clicked (left/top in % of the picture).
 * "Not a player" saves no name: the tracking drops that person, e.g. a ball fetcher it took for a player.
 */
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
  onSave: (name: string, notPlayer: boolean) => Promise<string | null>;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const above = top > 55; // keep the box on the picture
  async function save(notPlayer: boolean) {
    setSaving(true);
    setError(await onSave(notPlayer ? "" : name.trim(), notPlayer));
    setSaving(false);
  }
  return (
    <form
      className="name-box"
      style={{
        left: `${Math.min(Math.max(left, 18), 82)}%`,
        top: `${top}%`,
        transform: above ? "translate(-50%, calc(-100% - 10px))" : "translate(-50%, 10px)",
      }}
      onSubmit={(e) => {
        e.preventDefault();
        save(false);
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
        <button type="button" onClick={() => save(true)} disabled={saving} title="A ball fetcher, coach or anyone else">
          Not a player
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
          const label = `${tagLabel(t)} · ${formatTime(t.time_s)}`;
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

/**
 * Before analysis: click each player on a frame and name them, at five moments
 * spread across the match, so the tracking sees each of them at different times.
 */
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
  const moments = useMemo(() => labelMoments(durationS), [durationS]);
  const [t, setT] = useState(moments[0] ?? 0);
  const [size, setSize] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<[number, number] | null>(null);
  const img = useRef<HTMLImageElement>(null);
  const frameUrl = useMemo(() => api.frameUrl(matchId, t), [api, matchId, t]);
  const here = tags.filter((tag) => Math.abs(tag.time_s - t) < 0.5);
  const counts = labelCounts(tags, labelGap(durationS));
  const moment = moments.indexOf(t);
  const before = moments.filter((m) => m < t).at(-1);
  const after = moments.find((m) => m > t);
  // Everyone named on this frame: on to the next moment.
  const nextReady = new Set(here.filter((tag) => !tag.not_player).map((tag) => tag.name.toLowerCase())).size >= 4;

  function go(to: number) {
    setDraft(null);
    setT(to);
  }

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
        Click each of the four players and type their name, then go to the next moment and do it again. Naming
        them on {LABEL_GOAL} frames spread across the match shows the tracking how they look at different times,
        which keeps them apart. Click a ball fetcher or anyone else it picks up and choose Not a player. The slider
        picks any other frame, and you can also name players later, while watching.
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
              <span>{tagLabel(tag)}</span>
            </div>
          ))}
        {size && draft && (
          <NameBox
            left={(draft[0] / size[0]) * 100}
            top={(draft[1] / size[1]) * 100}
            suggestions={suggestions}
            onSave={async (name, notPlayer) => {
              const error = await onTag({
                time_s: t, x: draft[0], y: draft[1], name, player: null, not_player: notPlayer,
              });
              if (!error) setDraft(null);
              return error;
            }}
            onCancel={() => setDraft(null)}
          />
        )}
      </div>
      {moments.length > 1 && (
        <div className="row small" style={{ marginTop: 8 }}>
          <span className="muted grow">
            {moment >= 0 ? `Moment ${moment + 1} of ${moments.length}` : "A frame you picked"}
          </span>
          <button onClick={() => before !== undefined && go(before)} disabled={before === undefined}>
            Previous moment
          </button>
          <button
            className={nextReady ? "primary" : ""}
            onClick={() => after !== undefined && go(after)}
            disabled={after === undefined}
          >
            Next moment
          </button>
        </div>
      )}
      {durationS ? (
        <div className="row small" style={{ marginTop: 8 }}>
          <span className="muted">Frame at {formatTime(t)}</span>
          <input
            className="grow"
            type="range"
            min={0}
            max={Math.max(0, Math.floor(durationS - 1))}
            value={t}
            onChange={(e) => go(Number(e.target.value))}
            aria-label="Choose a frame where the players are easy to see"
          />
        </div>
      ) : null}
      <LabelProgress counts={counts} />
      <TagList tags={tags} onDelete={onDelete} />
      <div className="row" style={{ marginTop: 12 }}>
        <button className="primary" onClick={onDone}>
          Done
        </button>
      </div>
    </div>
  );
}

/** Each player's count towards {LABEL_GOAL} names, e.g. "Maya 3 of 5", and how many players are still unnamed. */
function LabelProgress({ counts }: { counts: { name: string; count: number }[] }) {
  const unnamed = Math.max(0, 4 - counts.length);
  return (
    <div className="row small" style={{ marginTop: 10, gap: 6 }} aria-label="How often each player is named">
      {counts.map(({ name, count }) => (
        <span key={name} className={`badge${count >= LABEL_GOAL ? " done" : ""}`}>
          {name} {Math.min(count, LABEL_GOAL)} of {LABEL_GOAL}
          {count >= LABEL_GOAL ? " ✓" : ""}
        </span>
      ))}
      {unnamed > 0 && (
        <span className="muted">
          {unnamed === 4 ? "No players named yet" : `${unnamed} more player${unnamed > 1 ? "s" : ""} to name`}
        </span>
      )}
    </div>
  );
}
