import { describe, expect, it } from "vitest";
import { lowerBound, nearest, toMatchTime, toVideoTime, type Segment } from "../lib/timeline";

// Two points: 100-130 s and 200-210 s of the match, back to back in the video.
const segments: Segment[] = [
  [100, 130, 0],
  [200, 210, 30.02],
];

describe("condensed match clock", () => {
  it("maps video time to match time", () => {
    expect(toMatchTime(segments, 0)).toBe(100);
    expect(toMatchTime(segments, 12.5)).toBe(112.5);
    expect(toMatchTime(segments, 30.02)).toBe(200);
    expect(toMatchTime(segments, 35.02)).toBeCloseTo(205);
    // The few frames between a point's nominal end and the next part stay on that point.
    expect(toMatchTime(segments, 30.01)).toBe(130);
  });

  it("maps match time to video time, snapping forward between points", () => {
    expect(toVideoTime(segments, 112.5)).toBe(12.5);
    expect(toVideoTime(segments, 50)).toBe(0);
    expect(toVideoTime(segments, 150)).toBe(30.02);
    expect(toVideoTime(segments, 205)).toBeCloseTo(35.02);
    expect(toVideoTime(segments, 300)).toBeNull();
  });

  it("is the identity for the full match", () => {
    expect(toMatchTime(null, 42)).toBe(42);
    expect(toVideoTime([], 42)).toBe(42);
  });
});

describe("sample lookup", () => {
  const t = [0, 0.1, 0.2, 0.5, 2.0];

  it("finds the closest sample within the gap", () => {
    expect(nearest(t, 0.14, 0.2)).toBe(1);
    expect(nearest(t, 0.16, 0.2)).toBe(2);
    expect(nearest(t, 1.2, 0.3)).toBe(-1);
    expect(nearest(t, 5, 0.3)).toBe(-1);
    expect(nearest([], 1, 1)).toBe(-1);
  });

  it("finds where a time would go", () => {
    expect(lowerBound(t, 0.2)).toBe(2);
    expect(lowerBound(t, 0.3)).toBe(3);
    expect(lowerBound(t, 9)).toBe(5);
  });
});
