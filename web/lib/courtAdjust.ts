// Nudging the court lines drawn over the video, in the browser only.
//
// The engine draws the court from the saved calibration. When those lines sit a
// little off the paint, the viewer can move them by hand: four handles on the
// court (a quadrilateral of known court points) are dragged onto the real
// lines, and every drawn line follows through the homography that takes the
// handles from where the calibration put them to where they were dropped.
//
// Nothing here is sent to the server: the saved calibration, and so the
// tracking and any analysis that is running, stay exactly as they were.

export type Pt = [number, number];
export type Mat3 = number[][];

/** A handle: a court point (metres) and where it sits in the source video (pixels). */
export type Anchor = { name: string; court: Pt; image: Pt };

/** What is kept for a match: where each handle was, and where it was moved to. */
export type CourtAdjustment = { size: Pt; from: Pt[]; to: Pt[] };

// Court points the handles can sit on (x across, y from the near wall), far pair first.
const FAR: [string, Pt][] = [
  ["far left corner", [0, 20]],
  ["far right corner", [10, 20]],
];
const NEAR_CHOICES: [string, Pt][][] = [
  [["near service right", [10, 3.05]], ["near service left", [0, 3.05]]],
  [["net right", [10, 10]], ["net left", [0, 10]]],
];

/** Maps a point through a 3x3 homography; null when it lands at infinity or behind the camera. */
export function apply(H: Mat3, [x, y]: Pt): Pt | null {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (!(Math.abs(w) > 1e-12)) return null;
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

/**
 * The homography taking each `src[i]` to `dst[i]`: exact for four points,
 * least squares for more. Null when the points are degenerate (three in a line).
 */
export function homography(src: Pt[], dst: Pt[]): Mat3 | null {
  if (src.length < 4 || src.length !== dst.length) return null;
  // Normalise both sets (centre, mean distance √2) so the solve is well conditioned in pixels.
  const [Ts, s] = normalise(src);
  const [Td, d] = normalise(dst);
  // Unknowns h0..h7 with h8 = 1; two equations per point, solved by normal equations.
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < s.length; i++) {
    const [x, y] = s[i];
    const [u, v] = d[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const AtA = Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => A.reduce((acc, row) => acc + row[r] * row[c], 0)),
  );
  const Atb = Array.from({ length: 8 }, (_, r) => A.reduce((acc, row, k) => acc + row[r] * b[k], 0));
  const h = solve(AtA, Atb);
  if (!h) return null;
  const Hn: Mat3 = [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
  const Tdi = invert(Td);
  if (!Tdi) return null;
  const H = mul(Tdi, mul(Hn, Ts));
  return H[2][2] ? H.map((row) => row.map((v) => v / H[2][2])) : null;
}

/**
 * The four handles for this calibration: the far corners, plus the near
 * service line ends when both are in the picture, otherwise the net ends.
 * Null when they can't be placed (the far corners are behind the camera).
 */
export function anchorsFor(courtToImage: Mat3, size: Pt): Anchor[] | null {
  const inside = ([u, v]: Pt) => u >= 0 && v >= 0 && u <= size[0] && v <= size[1];
  const place = ([name, court]: [string, Pt]): Anchor | null => {
    const image = apply(courtToImage, court);
    return image && inFront(courtToImage, court) ? { name, court, image } : null;
  };
  const far = FAR.map(place);
  if (far.some((a) => !a)) return null;
  for (const choice of NEAR_CHOICES) {
    const near = choice.map(place);
    if (near.every((a) => a && inside(a.image))) return [...far, ...near] as Anchor[];
  }
  const fallback = NEAR_CHOICES[NEAR_CHOICES.length - 1].map(place);
  return fallback.every(Boolean) ? ([...far, ...fallback] as Anchor[]) : null;
}

/**
 * The correction to draw with, or null to draw the lines as the engine made
 * them. An adjustment made for a different calibration (the handles no longer
 * start where it says) is ignored, so recalibrating never leaves stale lines.
 */
export function correction(adj: CourtAdjustment | null, anchors: Anchor[] | null, size: Pt): Mat3 | null {
  if (!adj || !anchors || adj.from.length !== anchors.length || adj.to.length !== anchors.length) return null;
  if (adj.size[0] !== size[0] || adj.size[1] !== size[1]) return null;
  const tol = Math.max(2, 0.002 * Math.max(size[0], size[1]));
  if (anchors.some((a, i) => Math.hypot(a.image[0] - adj.from[i][0], a.image[1] - adj.from[i][1]) > tol)) return null;
  if (adj.from.every((p, i) => p[0] === adj.to[i][0] && p[1] === adj.to[i][1])) return null;
  return homography(adj.from, adj.to);
}

/** The engine's polylines moved by the correction `G` (unchanged when it is null). */
export function adjustLines<L extends { points: Pt[] }>(lines: L[], G: Mat3 | null): L[] {
  if (!G) return lines;
  return lines.map((line) => ({
    ...line,
    points: line.points.map((p) => apply(G, p)).filter((p): p is Pt => p !== null),
  }));
}

/** True when the handles, in order, still make a convex quadrilateral (a drag must not turn it inside out). */
export function convex(quad: Pt[]): boolean {
  let sign = 0;
  for (let i = 0; i < quad.length; i++) {
    const [a, b, c] = [quad[i], quad[(i + 1) % quad.length], quad[(i + 2) % quad.length]];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cross) < 1e-9) return false;
    if (sign && Math.sign(cross) !== sign) return false;
    sign = Math.sign(cross);
  }
  return true;
}

const KEY = (matchId: string) => `padel-buddy:court-adjust:${matchId}`;

export function loadAdjustment(matchId: string): CourtAdjustment | null {
  try {
    const raw = window.localStorage.getItem(KEY(matchId));
    if (!raw) return null;
    const v = JSON.parse(raw) as CourtAdjustment;
    return Array.isArray(v?.from) && Array.isArray(v?.to) && Array.isArray(v?.size) ? v : null;
  } catch {
    return null;
  }
}

export function storeAdjustment(matchId: string, adj: CourtAdjustment | null) {
  try {
    if (adj) window.localStorage.setItem(KEY(matchId), JSON.stringify(adj));
    else window.localStorage.removeItem(KEY(matchId));
  } catch {
    // Private windows and blocked storage: the lines stay moved until the page is reloaded.
  }
}

function inFront(H: Mat3, [x, y]: Pt): boolean {
  // The camera sees the court with a positive w at a point in view; the centre of the court always is.
  const w = (p: Pt) => H[2][0] * p[0] + H[2][1] * p[1] + H[2][2];
  return Math.sign(w([x, y])) === Math.sign(w([5, 15]));
}

function normalise(pts: Pt[]): [Mat3, Pt[]] {
  const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
  const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  const mean = pts.reduce((a, p) => a + Math.hypot(p[0] - cx, p[1] - cy), 0) / pts.length || 1;
  const k = Math.SQRT2 / mean;
  const T: Mat3 = [
    [k, 0, -k * cx],
    [0, k, -k * cy],
    [0, 0, 1],
  ];
  return [T, pts.map(([x, y]) => [k * (x - cx), k * (y - cy)] as Pt)];
}

function mul(A: Mat3, B: Mat3): Mat3 {
  return A.map((row) => [0, 1, 2].map((c) => row[0] * B[0][c] + row[1] * B[1][c] + row[2] * B[2][c]));
}

function invert(M: Mat3): Mat3 | null {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

/** Gaussian elimination with partial pivoting; null for a singular system. */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-10) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}
