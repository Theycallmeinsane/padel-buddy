// Pointing at players: from a click on the picture to the box under it.

import type { Overlay } from "./api";
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
