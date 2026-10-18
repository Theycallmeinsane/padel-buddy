// Mapping between a video's own clock and the match clock, and lookups into
// the engine's per-player samples.

/** [start in the match, end in the match, start in the video], one per point. */
export type Segment = [number, number, number];

/** The match time showing at `videoTime`. No segments means the full match. */
export function toMatchTime(segments: Segment[] | null, videoTime: number): number {
  if (!segments?.length) return videoTime;
  let lo = 0;
  let hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (segments[mid][2] <= videoTime) lo = mid;
    else hi = mid - 1;
  }
  const [start, end, offset] = segments[lo];
  return Math.min(end, start + Math.max(0, videoTime - offset));
}

/**
 * Where match time `t` is in the video. Between points this is the start of
 * the next point; after the last one, null.
 */
export function toVideoTime(segments: Segment[] | null, t: number): number | null {
  if (!segments?.length) return t;
  for (const [start, end, offset] of segments) {
    if (t < start) return offset;
    if (t <= end) return offset + (t - start);
  }
  return null;
}

/** Index of the sample in sorted `times` closest to `t`, or -1 if none is within `maxGap`. */
export function nearest(times: number[], t: number, maxGap: number): number {
  if (!times.length) return -1;
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  const best = lo > 0 && t - times[lo - 1] < times[lo] - t ? lo - 1 : lo;
  return Math.abs(times[best] - t) <= maxGap ? best : -1;
}

/** First index with times[i] >= t. */
export function lowerBound(times: number[], t: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
