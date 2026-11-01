// Cutting clips out of the full match from times typed in, one clip per line.

import { parseTime } from "./fixes";

export type CutRangeIn = { start_s: number; end_s: number; name: string };
export type CutLineError = { line: number; text: string; message: string };

// "1:05 - 1:42 Great lob": a start, a dash (or "to"), an end, then an optional name.
const RANGE = /^(\d(?:[\d:.]*\d)?)\s*(?:-|–|—|to\b)\s*(\d(?:[\d:.]*\d)?)\s*(.*)$/i;

/**
 * Read clips typed one per line (or separated by commas) as "start - end name".
 * Times are "75", "1:15" or "1:02:03"; the name is optional. With the match's
 * length known, clips past its end are refused, and an end within a second
 * of it is taken as the end.
 */
export function parseCuts(text: string, durationS?: number | null): { ranges: CutRangeIn[]; errors: CutLineError[] } {
  const ranges: CutRangeIn[] = [];
  const errors: CutLineError[] = [];
  const lines = text.split(/[\n,;]/).map((l) => l.trim());
  lines.forEach((line, i) => {
    if (!line) return;
    const fail = (message: string) => errors.push({ line: i + 1, text: line, message });
    const m = RANGE.exec(line);
    if (!m) return fail("Type it as start - end, e.g. 1:05 - 1:42");
    const start = parseTime(m[1]);
    const end = parseTime(m[2]);
    if (start === null) return fail(`"${m[1]}" isn't a time: use 75, 1:15 or 1:02:03`);
    if (end === null) return fail(`"${m[2]}" isn't a time: use 75, 1:15 or 1:02:03`);
    if (end - start < 0.5) return fail("The end must come after the start");
    const known = durationS != null && durationS > 0;
    if (known && start >= durationS) return fail("It starts after the video ends");
    if (known && end > durationS + 1) return fail("It ends after the video ends");
    const name = m[3].replace(/^[\s:–—-]+/, "").trim().slice(0, 60);
    ranges.push({ start_s: start, end_s: known ? Math.min(end, durationS) : end, name });
  });
  return { ranges, errors };
}
