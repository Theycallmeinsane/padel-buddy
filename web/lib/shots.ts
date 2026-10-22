// Words for the engine's shots: kinds, how each kind went, and how a point ended.

import type { PointEnding, ShotKind, ShotStat } from "./api";

export const KIND_LABELS: Record<ShotKind, string> = {
  serve: "Serve",
  return: "Return",
  volley: "Volley",
  smash: "Smash",
  vibora: "Vibora",
  bandeja: "Bandeja",
  lob: "Lob",
  flat: "Flat drive",
  groundstroke: "Groundstroke",
  unknown: "Not sure",
};

export function kindLabel(kind: string): string {
  return KIND_LABELS[kind as ShotKind] ?? kind;
}

/** "80% of 10": kept in play or won outright, of the shots whose outcome is known; "–" when none is. */
export function keptInPlay(s: ShotStat): string {
  // A narrow cell wraps after the percentage, keeping "of 10" together.
  return s.success === null || !s.known ? "–" : `${Math.round(s.success * 100)}% of\u00a0${s.known}`;
}

/** "Strongest: Return, Volley. Work on: Vibora." with only the parts there are. */
export function focusLine(strengths: string[] = [], weaknesses: string[] = []): string {
  const parts = [];
  if (strengths.length) parts.push(`Strongest: ${strengths.map(kindLabel).join(", ")}.`);
  if (weaknesses.length) parts.push(`Work on: ${weaknesses.map(kindLabel).join(", ")}.`);
  return parts.join(" ");
}

/** For a report from before shot stats existed: how many of each kind, nothing about how they went. */
export function statsFromKinds(kinds: Partial<Record<ShotKind, number>>, errorsByKind: Record<string, number> = {}): ShotStat[] {
  return (Object.entries(kinds) as [ShotKind, number][])
    .filter(([k, n]) => k !== "unknown" && n > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([kind, shots]) => ({
      kind,
      shots,
      known: 0,
      in_play: 0,
      winners: 0,
      errors: errorsByKind[kind] ?? 0,
      success: null,
      rating: null,
    }));
}

/** "Winner by Maya", "Error by Ali (net)", or "" when how the point ended isn't known. */
export function endingLabel(p: PointEnding | undefined, names: Record<string, string> = {}): string {
  if (!p || p.ending === "unknown") return "";
  const who = p.last_player ? names[p.last_player] ?? p.last_player : sideLabel(p, p.last_team, names);
  const what = p.ending === "winner" ? "Winner" : "Error";
  const detail = p.ending === "error" && p.detail ? ` (${p.detail})` : "";
  return `${what}${who ? ` by ${who}` : ""}${detail}`;
}

/** Who played a point on one side: "Maya & Sam", else "the near side" or "the far side"; "" for no side. */
export function sideLabel(p: PointEnding, side: "A" | "B" | null, names: Record<string, string> = {}): string {
  if (!side) return "";
  const who = p.teams?.[side] ?? [];
  if (who.length) return who.map((pid) => names[pid] ?? pid).join(" & ");
  return side === "A" ? "the near side" : "the far side";
}
