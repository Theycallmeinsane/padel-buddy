import { describe, expect, it } from "vitest";
import type { Overlay } from "../lib/api";
import { playerAt, suggestions, toSource } from "../lib/tags";

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
});
