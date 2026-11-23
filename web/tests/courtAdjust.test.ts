import { describe, expect, it } from "vitest";
import {
  adjustLines,
  anchorsFor,
  apply,
  convex,
  correction,
  homography,
  type Mat3,
  type Pt,
} from "../lib/courtAdjust";

// A plausible camera behind the near wall: court metres to 1920x1080 pixels.
const court: Pt[] = [[0, 20], [10, 20], [10, 3.05], [0, 3.05]];
const image: Pt[] = [[700, 300], [1220, 300], [1700, 900], [220, 900]];
const H = homography(court, image)!;
const size: Pt = [1920, 1080];

describe("homography", () => {
  it("maps four points exactly, and points between them consistently", () => {
    court.forEach((c, i) => {
      const p = apply(H, c)!;
      expect(p[0]).toBeCloseTo(image[i][0], 6);
      expect(p[1]).toBeCloseTo(image[i][1], 6);
    });
    const net = apply(H, [5, 10])!;
    expect(net[0]).toBeCloseTo(960, 6); // symmetric court, centred camera
    expect(net[1]).toBeGreaterThan(300);
    expect(net[1]).toBeLessThan(900);
  });

  it("refuses three points in a line", () => {
    expect(homography([[0, 0], [1, 1], [2, 2], [3, 3]], image)).toBeNull();
  });
});

describe("anchors", () => {
  it("uses the far corners and near service ends when they are in the picture", () => {
    const a = anchorsFor(H, size)!;
    expect(a.map((x) => x.name)).toEqual(["far left corner", "far right corner", "near service right", "near service left"]);
    expect(a[2].image[0]).toBeCloseTo(1700, 6);
  });

  it("falls back to the net ends when the near service line runs off the picture", () => {
    const wide = homography(court, [[700, 300], [1220, 300], [2100, 1000], [-180, 1000]])!;
    expect(anchorsFor(wide, size)!.map((x) => x.name).slice(2)).toEqual(["net right", "net left"]);
  });
});

describe("correction", () => {
  const anchors = anchorsFor(H, size)!;
  const from = anchors.map((a) => a.image);

  it("is nothing until a handle moves", () => {
    expect(correction(null, anchors, size)).toBeNull();
    expect(correction({ size, from, to: from }, anchors, size)).toBeNull();
  });

  it("moves the lines so the handles land where they were dropped", () => {
    const to: Pt[] = from.map(([u, v], i) => (i === 0 ? [u - 12, v + 5] : [u, v]));
    const G = correction({ size, from, to }, anchors, size) as Mat3;
    const [line] = adjustLines([{ name: "far_wall", points: [from[0], from[1]] }], G);
    expect(line.name).toBe("far_wall");
    expect(line.points[0][0]).toBeCloseTo(from[0][0] - 12, 6);
    expect(line.points[0][1]).toBeCloseTo(from[0][1] + 5, 6);
    expect(line.points[1][0]).toBeCloseTo(from[1][0], 6);
  });

  it("is dropped after a recalibration moved where the handles start", () => {
    const to: Pt[] = from.map(([u, v]) => [u + 10, v]);
    const moved = anchors.map((a) => ({ ...a, image: [a.image[0] + 40, a.image[1]] as Pt }));
    expect(correction({ size, from, to }, moved, size)).toBeNull();
    expect(correction({ size: [1280, 720], from, to }, anchors, size)).toBeNull();
  });
});

describe("convex", () => {
  it("accepts the court and refuses a handle dragged across it", () => {
    expect(convex(image)).toBe(true);
    expect(convex([[700, 300], [1220, 300], [600, 200], [220, 900]])).toBe(false);
  });
});
