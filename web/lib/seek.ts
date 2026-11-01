/** How far the skip buttons under each video jump, in seconds. */
export const SKIP_S = 10;

/**
 * Where a relative skip lands: `delta` seconds from `current`, never before the start and, once the
 * length is known, never past the end. Before the metadata loads the duration is NaN, and a stream
 * of unknown length reports Infinity, so only the start is a hard limit then.
 */
export function skipTarget(current: number, delta: number, duration: number): number {
  const from = Number.isFinite(current) ? current : 0;
  const to = Math.max(0, from + delta);
  return Number.isFinite(duration) && duration >= 0 ? Math.min(to, duration) : to;
}

/** Move a video `delta` seconds from where it is now. Returns the new time. */
export function seekBy(video: Pick<HTMLVideoElement, "currentTime" | "duration">, delta: number): number {
  const to = skipTarget(video.currentTime, delta, video.duration);
  video.currentTime = to;
  return to;
}
