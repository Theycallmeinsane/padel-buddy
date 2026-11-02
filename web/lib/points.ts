// Where each point's download starts and ends, and moving that by hand when the split was off.

import type { ClipsSummary, Rally } from "./api";

/** The download runs this long past the end the analysis found, which tends to be early (as in the API). */
export const POINT_END_PAD_S = 5;
/** The shortest a point can be trimmed to. */
export const MIN_POINT_S = 1;
/** How far around the point the trim bar reaches. */
export const TRIM_REACH_S = 20;

export type PointEdits = Record<string, [number, number]>;

/**
 * What the point's download covers: the times set by hand, or else the stretch
 * of the condensed match holding the point, running on POINT_END_PAD_S past it.
 * Mirrors `_original_spans` in the API.
 */
export function pointSpan(
  rally: Rally,
  clips: Pick<ClipsSummary, "condensed_segments"> | null,
  edits: PointEdits = {},
  durationS?: number | null,
): { start: number; end: number; edited: boolean } {
  const edit = edits[String(rally.index)];
  if (edit) return { start: edit[0], end: edit[1], edited: true };
  const seg = (clips?.condensed_segments ?? []).find(([a, b]) => a <= rally.start_s && rally.start_s <= b);
  const [a, b] = seg ? [seg[0], seg[1]] : [Math.max(0, rally.start_s - 2), rally.end_s + 2];
  const end = b + POINT_END_PAD_S;
  return { start: a, end: durationS ? Math.min(end, durationS) : end, edited: false };
}

/** The stretch of the match the trim bar shows around a point. */
export function trimWindow(start: number, end: number, durationS?: number | null): [number, number] {
  const lo = Math.max(0, start - TRIM_REACH_S);
  const hi = end + TRIM_REACH_S;
  return [lo, durationS ? Math.min(hi, durationS) : hi];
}

/**
 * Move one end of the point to `t`, keeping it inside the window and at least
 * MIN_POINT_S long; the other end stays put.
 */
export function moveEdge(
  span: [number, number],
  edge: "start" | "end",
  t: number,
  window: [number, number],
): [number, number] {
  const [start, end] = span;
  const [lo, hi] = window;
  if (edge === "start") return [round(Math.min(Math.max(t, lo), end - MIN_POINT_S)), end];
  return [start, round(Math.max(Math.min(t, hi), start + MIN_POINT_S))];
}

function round(t: number): number {
  return Math.round(t * 10) / 10;
}
