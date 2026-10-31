import { describe, expect, it } from "vitest";
import type { Fix } from "../lib/api";
import { fixProblem, fixSummary, parseTime, playerName } from "../lib/fixes";

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

describe("putting a player right by hand", () => {
  it("reads match times as typed", () => {
    expect(parseTime("12:30")).toBe(750);
    expect(parseTime(" 1:02:03 ")).toBe(3723);
    expect(parseTime("75")).toBe(75);
    expect(parseTime("0:05.5")).toBe(5.5);
    for (const bad of ["", "abc", "1:75", "1::2", "-3", "1:2:3:4"]) expect(parseTime(bad)).toBeNull();
  });

  it("names players, by number when they have no name", () => {
    expect(playerName({ P1: "Maya" }, "P1")).toBe("Maya");
    expect(playerName({ P1: "Maya" }, "P3")).toBe("P3");
  });

  it("says what a fix does", () => {
    const names = { P1: "Maya", P3: "Ali" };
    const fix: Fix = { id: "a", start_s: 750, end_s: null, player: "P1", becomes: "P3" };
    expect(fixSummary(fix, names, clock)).toBe("From 12:30 to the end: Maya is really Ali");
    expect(fixSummary({ ...fix, end_s: 800, becomes: "P4" }, names, clock)).toBe(
      "From 12:30 to 13:20: Maya is really P4",
    );
  });

  it("refuses a fix that can't be saved", () => {
    expect(fixProblem(750, null, "P1", "P3")).toBeNull();
    expect(fixProblem(750, 800, "P1", "P3")).toBeNull();
    expect(fixProblem(null, null, "P1", "P3")).toMatch(/start/);
    expect(fixProblem(750, null, "P1", "P1")).toMatch(/someone else/);
    expect(fixProblem(750, 700, "P1", "P3")).toMatch(/after the start/);
    expect(fixProblem(750, null, "P1", "")).toMatch(/Choose/);
  });
});
