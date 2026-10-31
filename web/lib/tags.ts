// Pointing at players: from a click on the picture to the box under it.

import type { Overlay, Tag } from "./api";
import { nearest } from "./timeline";

/**
 * Where a click lands in source video pixels. The picture is letterboxed
 * inside an element of `boxW` x `boxH` CSS pixels and shows a `videoW` x
 * `videoH` video, which may be smaller than the `size` the engine measured in
 * (the condensed match is 720p). Null outside the picture.
 */
export function toSource(
  clickX: number,
  clickY: number,
  boxW: number,
  boxH: number,
  videoW: number,
  videoH: number,
  size: [number, number],
): [number, number] | null {
  const fit = Math.min(boxW / videoW, boxH / videoH);
  const pw = videoW * fit;
  const ph = videoH * fit;
  const x = clickX - (boxW - pw) / 2;
  const y = clickY - (boxH - ph) / 2;
  if (x < 0 || y < 0 || x > pw || y > ph) return null;
  return [(x / pw) * size[0], (y / ph) * size[1]];
}

/** The player whose box is under (x, y) at match time `t`; the smallest box when they overlap. */
export function playerAt(overlay: Overlay, t: number, x: number, y: number, maxGap: number): string | null {
  let best: string | null = null;
  let bestArea = Infinity;
  for (const [pid, p] of Object.entries(overlay.players)) {
    const i = nearest(p.t, t, maxGap);
    if (i < 0) continue;
    const [x1, y1, x2, y2] = p.box.slice(4 * i, 4 * i + 4);
    const area = (x2 - x1) * (y2 - y1);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && area < bestArea) {
      best = pid;
      bestArea = area;
    }
  }
  return best;
}

/**
 * Names to show while a match is analysed: what the tags say so far (`tagged`,
 * from the engine) wins, and a saved name a tag has given to someone else is dropped.
 */
export function liveNames(saved: Record<string, string>, tagged: Record<string, string>): Record<string, string> {
  const given = new Set(Object.values(tagged).map((n) => n.toLowerCase()));
  const kept = Object.entries(saved).filter(([, n]) => !given.has(n.toLowerCase()));
  return { ...Object.fromEntries(kept), ...tagged };
}

/** Names to suggest: this match's first, then everyone named before, each once. */
export function suggestions(names: Record<string, string>, people: string[]): string[] {
  const seen = new Set<string>();
  const out = [];
  for (const n of [...Object.values(names), ...people]) {
    if (seen.has(n.toLowerCase())) continue;
    seen.add(n.toLowerCase());
    out.push(n);
  }
  return out;
}

/** How many frames each player is named on before analysing, so the tracking keeps them apart all match. */
export const LABEL_GOAL = 5;

/** Moments to name the players at, spread evenly across the match: 10%, 30%, 50%, 70% and 90% of the way in. */
export function labelMoments(durationS: number | null, n = LABEL_GOAL): number[] {
  if (!durationS) return [];
  const last = Math.max(0, Math.floor(durationS - 1));
  const out = new Set<number>();
  for (let i = 0; i < n; i++) out.add(Math.min(last, Math.floor((durationS * (2 * i + 1)) / (2 * n))));
  return [...out];
}

/** How far apart two names must be to count twice: 10 s, less in a match too short to fit the moments. */
export function labelGap(durationS: number | null): number {
  return Math.min(10, (durationS ?? 0) / (2 * LABEL_GOAL));
}

export type LabelCount = { name: string; count: number };

/**
 * How many times each player is named, counting only names at least `gap`
 * seconds apart: the same moment twice shows the tracking nothing new.
 * Not-a-player tags and tags that found nobody to name don't count.
 */
export function labelCounts(tags: Tag[], gap: number): LabelCount[] {
  const byName = new Map<string, { name: string; times: number[] }>();
  for (const tag of tags) {
    if (tag.not_player || !tag.name || tag.applied === false) continue;
    const key = tag.name.toLowerCase();
    const entry = byName.get(key) ?? { name: tag.name, times: [] };
    entry.times.push(tag.time_s);
    byName.set(key, entry);
  }
  return [...byName.values()].map(({ name, times }) => {
    let count = 0;
    let last = -Infinity;
    for (const t of times.sort((a, b) => a - b)) {
      if (t - last < gap) continue;
      count++;
      last = t;
    }
    return { name, count };
  });
}

/** What is left to name before all four players reach the goal, e.g. ["Maya 3 of 5", "name 2 more players"]. */
export function labelsToDo(counts: LabelCount[], goal = LABEL_GOAL): string[] {
  const todo = counts.filter((c) => c.count < goal).map((c) => `${c.name} ${c.count} of ${goal}`);
  const missing = Math.max(0, 4 - counts.length);
  if (missing === 4) todo.push("name all four players");
  else if (missing) todo.push(`name ${missing} more player${missing > 1 ? "s" : ""}`);
  return todo;
}
