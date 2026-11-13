import { describe, expect, it } from "vitest";
import { cameraAdvice, checkFrames, checksAt, defaultCheckStart, framesDone, percent } from "../lib/checks";

const check = (time_s: number, name = "Maya") => ({ id: `${time_s}${name}`, time_s, x: 0, y: 0, name });

describe("checking the tracking", () => {
  it("starts a fifth of the way in, leaving room for ten minutes", () => {
    expect(defaultCheckStart(5000)).toBe(1000);
    expect(defaultCheckStart(700)).toBe(102);
    expect(defaultCheckStart(300)).toBe(0);
    expect(defaultCheckStart(null)).toBe(0);
  });

  it("checks a frame every 3 s, 200 of them, never past the end", () => {
    const frames = checkFrames(60, 5000);
    expect(frames.length).toBe(200);
    expect(frames.slice(0, 3)).toEqual([60, 63, 66]);
    expect(checkFrames(0, 10)).toEqual([0, 3, 6, 9]);
  });

  it("counts a frame done once all four are checked on it", () => {
    const checks = [check(3), check(3, "Sara"), check(3, "Ali"), check(3, "Noor"), check(6)];
    expect(checksAt(checks, 3).length).toBe(4);
    expect(framesDone(checks, [3, 6, 9])).toBe(1);
  });

  it("shows shares as whole percents", () => {
    expect(percent(0.8234)).toBe("82%");
    expect(percent(null)).toBe("–");
  });
});

describe("camera height", () => {
  it("advises mounting higher below 2 m only", () => {
    expect(cameraAdvice(1.28)).toContain("about 1.3 m high");
    expect(cameraAdvice(2.6)).toBeNull();
    expect(cameraAdvice(null)).toBeNull();
  });
});
