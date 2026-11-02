import { describe, expect, it } from "vitest";
import type { Rally } from "../lib/api";
import { moveEdge, pointSpan, trimWindow } from "../lib/points";

const rally = (index: number, start_s: number, end_s: number): Rally => ({
  index, start_s, end_s, hits: 6, peak_excess_db: 10, movement_mps: null, score: 1, clip: `points/point_00${index + 1}.mp4`,
});
const clips = { condensed_segments: [[8, 22, 0], [93, 100, 14]] as [number, number, number][] };

describe("a point's download", () => {
  it("runs 5 s past the end the split found", () => {
    expect(pointSpan(rally(0, 10, 20), clips)).toEqual({ start: 8, end: 27, edited: false });
    expect(pointSpan(rally(0, 10, 20), null)).toEqual({ start: 8, end: 27, edited: false }); // before the segments
  });

  it("stops at the end of the video", () => {
    expect(pointSpan(rally(1, 95, 98), clips, {}, 102)).toEqual({ start: 93, end: 102, edited: false });
  });

  it("uses the times set by hand as they are", () => {
    expect(pointSpan(rally(0, 10, 20), clips, { "0": [9, 31.5] })).toEqual({ start: 9, end: 31.5, edited: true });
    expect(pointSpan(rally(1, 95, 98), clips, { "0": [9, 31.5] }).edited).toBe(false);
  });
});

describe("dragging a point's start and end", () => {
  it("shows 20 s either side, inside the video", () => {
    expect(trimWindow(30, 50, 600)).toEqual([10, 70]);
    expect(trimWindow(5, 590, 600)).toEqual([0, 600]);
    expect(trimWindow(5, 50, null)).toEqual([0, 70]);
  });

  it("moves one end and leaves the other", () => {
    expect(moveEdge([30, 50], "start", 27.33, [10, 70])).toEqual([27.3, 50]);
    expect(moveEdge([30, 50], "end", 55, [10, 70])).toEqual([30, 55]);
  });

  it("keeps the point at least a second long and inside the window", () => {
    expect(moveEdge([30, 50], "start", 60, [10, 70])).toEqual([49, 50]);
    expect(moveEdge([30, 50], "end", 10, [10, 70])).toEqual([30, 31]);
    expect(moveEdge([30, 50], "start", 2, [10, 70])).toEqual([10, 50]);
    expect(moveEdge([30, 50], "end", 99, [10, 70])).toEqual([30, 70]);
  });
});
