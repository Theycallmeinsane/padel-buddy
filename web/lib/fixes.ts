// Putting who is who right by hand: the fallback when tags can't settle a mix-up.

import type { Fix } from "./api";

export const PLAYERS = ["P1", "P2", "P3", "P4"] as const;

/** A player's name, or their number when they have none. */
export function playerName(names: Record<string, string>, pid: string): string {
  return names[pid] || pid;
}

/** Match time typed as "1:02:03", "12:30" or "75" (seconds); null when it isn't one. */
export function parseTime(text: string): number | null {
  const parts = text.trim().split(":");
  if (!parts.length || parts.length > 3 || parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) return null;
  const nums = parts.map(Number);
  if (nums.slice(1).some((n) => n >= 60)) return null;
  return nums.reduce((total, n) => total * 60 + n, 0);
}

/** What a fix says, e.g. "From 12:30 to the end: Maya is really Ali". */
export function fixSummary(fix: Fix, names: Record<string, string>, clock: (s: number) => string): string {
  const until = fix.end_s === null ? "the end" : clock(fix.end_s);
  return `From ${clock(fix.start_s)} to ${until}: ${playerName(names, fix.player)} is really ${playerName(names, fix.becomes)}`;
}

/** The problem with a fix about to be saved, or null when it can be saved. */
export function fixProblem(start: number | null, end: number | null, player: string, becomes: string): string | null {
  if (start === null) return "Type the start as minutes:seconds, e.g. 12:30";
  if (!player || !becomes) return "Choose who is shown and who it really is";
  if (player === becomes) return "Choose someone else: that is who is shown there already";
  if (end !== null && end <= start) return "The end must come after the start";
  return null;
}
