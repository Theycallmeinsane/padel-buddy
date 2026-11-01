import { describe, expect, it } from "vitest";
import { parseCuts } from "../lib/cuts";

describe("cutting clips from typed times", () => {
  it("reads one clip per line, with an optional name", () => {
    const { ranges, errors } = parseCuts("1:05 - 1:42 Great lob\n12:30-13:10\n\n  1:02:03 to 1:02:30: Smash  ");
    expect(errors).toEqual([]);
    expect(ranges).toEqual([
      { start_s: 65, end_s: 102, name: "Great lob" },
      { start_s: 750, end_s: 790, name: "" },
      { start_s: 3723, end_s: 3750, name: "Smash" },
    ]);
  });

  it("takes seconds, decimals, other dashes and commas between clips", () => {
    const { ranges, errors } = parseCuts("75-90, 0:05.5 – 0:10; 100—110 last one");
    expect(errors).toEqual([]);
    expect(ranges.map((r) => [r.start_s, r.end_s])).toEqual([
      [75, 90],
      [5.5, 10],
      [100, 110],
    ]);
    expect(ranges[2].name).toBe("last one");
  });

  it("says which line is wrong and why, and keeps the rest", () => {
    const { ranges, errors } = parseCuts("1:00 - 1:30\nhello\n2:00 - 1:50\n1:75 - 2:00\n3:00 - 3:00.2");
    expect(ranges).toEqual([{ start_s: 60, end_s: 90, name: "" }]);
    expect(errors.map((e) => [e.line, e.message])).toEqual([
      [2, "Type it as start - end, e.g. 1:05 - 1:42"],
      [3, "The end must come after the start"],
      [4, `"1:75" isn't a time: use 75, 1:15 or 1:02:03`],
      [5, "The end must come after the start"],
    ]);
  });

  it("checks clips against the video's length", () => {
    const { ranges, errors } = parseCuts("9:50 - 10:00.5\n10:01 - 10:30\n9:00 - 11:00", 600);
    expect(ranges).toEqual([{ start_s: 590, end_s: 600, name: "" }]); // a rounded-up end is the end
    expect(errors.map((e) => e.message)).toEqual(["It starts after the video ends", "It ends after the video ends"]);
    expect(parseCuts("9:00 - 11:00", null).ranges).toEqual([{ start_s: 540, end_s: 660, name: "" }]);
  });

  it("finds nothing in an empty box", () => {
    expect(parseCuts("  \n\n")).toEqual({ ranges: [], errors: [] });
  });
});
