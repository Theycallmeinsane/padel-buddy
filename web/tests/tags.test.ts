import { describe, expect, it } from "vitest";
import type { Overlay, Tag } from "../lib/api";
import { labelCounts, labelGap, labelMoments, labelsToDo, liveNames, playerAt, suggestions, toSource } from "../lib/tags";

const overlay = {
  size: [1920, 1080],
  players: {
    P1: { team: "A", t: [10, 10.1], box: [100, 500, 300, 1000, 110, 500, 310, 1000], xy: [] },
    P3: { team: "B", t: [10], box: [180, 520, 240, 700], xy: [] }, // further away, seen through P1
  },
} as unknown as Overlay;

describe("tagging a player", () => {
  it("maps a click on a letterboxed, smaller video to source pixels", () => {
    // 1280x720 video in a 1000x1000 box: 562.5 px tall, centred.
    const at = toSource(500, 500, 1000, 1000, 1280, 720, [1920, 1080]);
    expect(at![0]).toBeCloseTo(960);
    expect(at![1]).toBeCloseTo(540);
    expect(toSource(500, 100, 1000, 1000, 1280, 720, [1920, 1080])).toBeNull(); // on the black bar
  });

  it("finds the box under the click, the smaller one where two overlap", () => {
    expect(playerAt(overlay, 10, 150, 900, 0.3)).toBe("P1");
    expect(playerAt(overlay, 10, 200, 600, 0.3)).toBe("P3");
    expect(playerAt(overlay, 10.1, 305, 900, 0.3)).toBe("P1"); // the box moved
    expect(playerAt(overlay, 10, 1000, 600, 0.3)).toBeNull();
    expect(playerAt(overlay, 20, 150, 900, 0.3)).toBeNull(); // nobody tracked then
  });

  it("suggests this match's names first, each once", () => {
    expect(suggestions({ P1: "Maya", P2: "Ali" }, ["ali", "Sam", "Maya"])).toEqual(["Maya", "Ali", "Sam"]);
  });

  it("shows the names tags gave so far while analysing, moving a saved name if a tag says so", () => {
    expect(liveNames({ P1: "Maya", P2: "Ali" }, { P3: "maya" })).toEqual({ P2: "Ali", P3: "maya" });
    expect(liveNames({ P1: "Maya" }, {})).toEqual({ P1: "Maya" });
  });
});

const tag = (name: string, time_s: number, extra: Partial<Tag> = {}): Tag => ({
  id: `${name}${time_s}`, time_s, x: 0, y: 0, name, player: null, applied: null, ...extra,
});

describe("naming each player five times", () => {
  it("offers five moments spread across the match", () => {
    expect(labelMoments(600)).toEqual([60, 180, 300, 420, 540]);
    expect(labelMoments(3)).toEqual([0, 1, 2]); // too short for five different frames
    expect(labelMoments(null)).toEqual([]);
  });

  it("counts names at least 10 s apart, fewer in a short match", () => {
    expect(labelGap(600)).toBe(10);
    expect(labelGap(50)).toBe(5);
    const tags = [
      tag("Maya", 60), tag("maya", 64), tag("Maya", 180), tag("Maya", 190),
      tag("Ali", 60), tag("", 60, { not_player: true }), tag("Sam", 300, { applied: false }),
    ];
    expect(labelCounts(tags, 10)).toEqual([{ name: "Maya", count: 3 }, { name: "Ali", count: 1 }]);
  });

  it("says what is left before analysing", () => {
    expect(labelsToDo([])).toEqual(["name all four players"]);
    expect(labelsToDo([{ name: "Maya", count: 3 }, { name: "Ali", count: 5 }])).toEqual([
      "Maya 3 of 5", "name 2 more players",
    ]);
    const done = ["Maya", "Ali", "Sam", "Noor"].map((name) => ({ name, count: 5 }));
    expect(labelsToDo(done)).toEqual([]);
  });
});
