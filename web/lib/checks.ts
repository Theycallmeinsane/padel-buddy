// Checking the tracking by hand: frames to check and what was checked on each.

import type { Check } from "./api";

/** Seconds between checked frames, and how many: 200 frames 3 s apart cover 10 minutes. */
export const CHECK_EVERY_S = 3;
export const CHECK_FRAMES = 200;

/** Where checking starts by default: a fifth of the way in, past the warm-up, leaving room for 10 minutes. */
export function defaultCheckStart(durationS: number | null): number {
  if (!durationS) return 0;
  const span = CHECK_EVERY_S * (CHECK_FRAMES - 1);
  return Math.max(0, Math.min(Math.floor(durationS * 0.2), Math.floor(durationS - span - 1)));
}

/** The frames to check from `start`, every CHECK_EVERY_S seconds, all inside the match. */
export function checkFrames(start: number, durationS: number | null): number[] {
  const out = [];
  for (let i = 0; i < CHECK_FRAMES; i++) {
    const t = start + i * CHECK_EVERY_S;
    if (durationS && t > durationS - 0.5) break;
    out.push(t);
  }
  return out;
}

/** Checks made on the frame at `t`. */
export function checksAt(checks: Check[], t: number): Check[] {
  return checks.filter((c) => Math.abs(c.time_s - t) < 0.05);
}

/** How many of `frames` have at least `perFrame` checks. */
export function framesDone(checks: Check[], frames: number[], perFrame = 4): number {
  return frames.filter((t) => checksAt(checks, t).length >= perFrame).length;
}

/** A share as a whole percent, or a dash when there is nothing to share. */
export function percent(v: number | null): string {
  return v === null ? "–" : `${Math.round(v * 100)}%`;
}

/** Below this the net hides the far players' legs, which is what tells players in similar shirts apart. */
export const LOW_CAMERA_M = 2;

/** Advice when the camera is mounted low, or null when its height is fine or unknown. */
export function cameraAdvice(heightM: number | null | undefined): string | null {
  if (heightM == null || heightM >= LOW_CAMERA_M) return null;
  return (
    `The camera is about ${heightM.toFixed(1)} m high, so the net hides the far players' legs and the tracking ` +
    "mixes them up more. For future matches, mount it 2.5 to 3 m up behind the back glass."
  );
}
