import { describe, expect, it } from "vitest";
import { byNumber, pairLabel, withNames, type PointEnding, type ShotStat } from "../lib/api";
import { endingLabel, focusLine, keptInPlay, kindLabel, statsFromKinds } from "../lib/shots";

const stat = (over: Partial<ShotStat>): ShotStat => ({
  kind: "volley",
  shots: 10,
  known: 10,
  in_play: 6,
  winners: 2,
  errors: 2,
  success: 0.8,
  rating: "solid",
  ...over,
});

describe("shot stats in words", () => {
  it("names every kind, and passes unknown ones through", () => {
    expect(kindLabel("vibora")).toBe("Vibora");
    expect(kindLabel("flat")).toBe("Flat drive");
    expect(kindLabel("tweener")).toBe("tweener");
  });

  it("says how often a kind was kept in play", () => {
    expect(keptInPlay(stat({}))).toBe("80% of\u00a010");
    expect(keptInPlay(stat({ known: 0, success: null }))).toBe("–");
  });

  it("sums up strengths and weaknesses, leaving out what's missing", () => {
    expect(focusLine(["return", "volley"], ["vibora"])).toBe("Strongest: Return, Volley. Work on: Vibora.");
    expect(focusLine([], ["flat"])).toBe("Work on: Flat drive.");
    expect(focusLine(["smash"])).toBe("Strongest: Smash.");
    expect(focusLine()).toBe("");
  });

  it("falls back to counts for an older report", () => {
    const out = statsFromKinds({ volley: 3, serve: 7, unknown: 4 }, { volley: 1 });
    expect(out.map((s) => [s.kind, s.shots, s.errors, s.success])).toEqual([
      ["serve", 7, 0, null],
      ["volley", 3, 1, null],
    ]);
  });
});

describe("how a point ended", () => {
  const point = (over: Partial<PointEnding>): PointEnding => ({
    rally: 0,
    start_s: 10,
    end_s: 20,
    shots: 6,
    last_player: "P1",
    last_team: "A",
    ending: "winner",
    detail: null,
    won_by: "A",
    ...over,
  });
  const names = { P1: "Maya", P2: "Ali" };

  it("names the player who finished it", () => {
    expect(endingLabel(point({}), names)).toBe("Winner by Maya");
    expect(endingLabel(point({ ending: "error", last_player: "P2", detail: "net", won_by: "B" }), names)).toBe(
      "Error by Ali (net)",
    );
    expect(endingLabel(point({ last_player: "P3", last_team: "B", won_by: "B" }), names)).toBe("Winner by P3");
    expect(endingLabel(point({ last_player: null }), names)).toBe("Winner by the near side");
    // Teams change point by point: whoever played it on that side.
    expect(endingLabel(point({ last_player: null, teams: { A: ["P1", "P5"], B: ["P2", "P3"] } }), names)).toBe(
      "Winner by Maya & P5",
    );
  });

  it("says nothing when it isn't known", () => {
    expect(endingLabel(point({ ending: "unknown", won_by: null }), names)).toBe("");
    expect(endingLabel(undefined, names)).toBe("");
  });
});

describe("names in engine text", () => {
  it("swaps numbers and pairs for names, P10 included", () => {
    const names = { P1: "Maya", P3: "Sam", P10: "Kim" };
    expect(withNames("P1+P3 played split; P10 and P1 lobbed; P2 too", names)).toBe(
      "Maya & Sam played split; Kim and Maya lobbed; P2 too",
    );
    expect(pairLabel("P2+P10", names)).toBe("P2 & Kim");
    expect(["P10", "P2", "P1"].sort(byNumber)).toEqual(["P1", "P2", "P10"]);
  });
});
