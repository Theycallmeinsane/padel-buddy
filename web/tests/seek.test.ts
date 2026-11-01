import { describe, expect, it } from "vitest";
import { seekBy, skipTarget } from "../lib/seek";

describe("10 second skip", () => {
  it("moves relative to the current time", () => {
    expect(skipTarget(30, 10, 120)).toBe(40);
    expect(skipTarget(30, -10, 120)).toBe(20);
    expect(skipTarget(12.5, 10, 120)).toBe(22.5);
  });

  it("stops at the start", () => {
    expect(skipTarget(4, -10, 120)).toBe(0);
    expect(skipTarget(0, -10, 120)).toBe(0);
  });

  it("stops at the end", () => {
    expect(skipTarget(115, 10, 120)).toBe(120);
    expect(skipTarget(120, 10, 120)).toBe(120);
  });

  it("only limits the start while the length is unknown", () => {
    expect(skipTarget(5, 10, NaN)).toBe(15);
    expect(skipTarget(5, 10, Infinity)).toBe(15);
    expect(skipTarget(5, -10, NaN)).toBe(0);
    expect(skipTarget(NaN, 10, 120)).toBe(10);
  });

  it("sets the video's time", () => {
    const video = { currentTime: 50, duration: 55 };
    expect(seekBy(video, 10)).toBe(55);
    expect(video.currentTime).toBe(55);
    seekBy(video, -10);
    expect(video.currentTime).toBe(45);
  });
});
