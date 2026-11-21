"""Find the court in a frame by itself, from its white lines and floor edges.

Clicking the far corners is the weak spot of a hand calibration: from a phone
behind the near wall the far half of the court is a thin strip, and a click
a few pixels off moves a far point by a metre. A painted line, though, is
hundreds of pixels long, and a straight line fitted through all of its
pixels sits far more precisely than any single click.

How it works:

1. The per-pixel median of a few frames removes the players.
2. The floor is the colour found around the bottom middle of the frame (or
   higher up, when the camera stands back and sees the surround first); its
   left and right edges are where the side walls stand (x = 0 and 10 m), and
   its top edge is the far wall (y = 20 m).
3. Painted lines are thin strokes brighter and paler than the floor around
   them; straight lines are fitted through them.
4. Each way of naming two of those lines as the near service line and the
   centre line is tried. Together with the side edges, they fix the camera,
   because a phone camera has square pixels and its lens axis through the
   middle of the picture: that leaves seven unknowns (focal length, three
   angles, three position coordinates), and those four lines give seven
   equations. The far end of the court then follows from geometry.
5. Except when the camera is square to the court: then a near camera with a
   wide lens and a distant one zoomed in see the near lines alike. So the
   camera's distance is stepped through, and the far end decides: the net's
   white tape (a known 0.88 m up), the far wall and the far service line.
6. The best fit is polished against every pixel of the lines, edges and net
   tape it explains.

A fit that leaves lines unexplained, or puts the camera somewhere a camera
can't be, is rejected with the reason, so the user can click the points
instead.

Assumptions: the camera stands behind one back wall and looks down the court,
the floor is one colour that differs from what lies beyond the side walls,
and the painted lines are white.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import permutations

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import court
from .calibration import Calibration, CalibrationError

WORK_SIZE = 1280  # long side, in pixels, that detection runs at

# Court lines in homogeneous form (a, b, c): a*x + b*y + c = 0, in metres.
MODEL_LINES: dict[str, np.ndarray] = {
    "near_service_line": np.array([0.0, 1.0, -court.NEAR_SERVICE_Y]),
    "far_service_line": np.array([0.0, 1.0, -court.FAR_SERVICE_Y]),
    "near_center_line": np.array([1.0, 0.0, -court.CENTER_X]),
    "far_center_line": np.array([1.0, 0.0, -court.CENTER_X]),
    "left_side": np.array([1.0, 0.0, 0.0]),
    "right_side": np.array([1.0, 0.0, -court.WIDTH]),
    "far_wall": np.array([0.0, 1.0, -court.LENGTH]),
}
# Where each line runs, in metres: the painted extent, or the wall's length.
EXTENT: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_service_line": ((0.0, court.NEAR_SERVICE_Y), (court.WIDTH, court.NEAR_SERVICE_Y)),
    "far_service_line": ((0.0, court.FAR_SERVICE_Y), (court.WIDTH, court.FAR_SERVICE_Y)),
    "near_center_line": ((court.CENTER_X, court.NEAR_SERVICE_Y - court.CENTER_LINE_OVERHANG), (court.CENTER_X, court.NET_Y)),
    "far_center_line": ((court.CENTER_X, court.NET_Y), (court.CENTER_X, court.FAR_SERVICE_Y + court.CENTER_LINE_OVERHANG)),
    "left_side": ((0.0, 0.0), (0.0, court.LENGTH)),
    "right_side": ((court.WIDTH, 0.0), (court.WIDTH, court.LENGTH)),
    "far_wall": ((0.0, court.LENGTH), (court.WIDTH, court.LENGTH)),
}
PAINTED = ("near_service_line", "near_center_line", "far_service_line", "far_center_line")
# The net's white top tape, 0.88 m high in the middle and 0.92 m at the posts
# (FIP rules); the middle of the 5 cm band is what shows. It is the clearest
# thing in the far half of the frame, and since it isn't on the floor it pins
# down how far away the camera is when the floor lines alone can't.
NET_TAPE_M = (0.88 - 0.025, 0.04)  # height of the band's middle at the centre, extra height at the posts
FAR_CUES = ("far_service_line", "far_wall", "net_tape")
MIN_FAR_SUPPORT = 0.5  # at least one far cue must match this well, or the far end is a guess
EDGES = ("left_side", "right_side", "far_wall")

# A fit is kept only when all of these hold.
MIN_SUPPORT = {"near_service_line": 0.6, "near_center_line": 0.5}  # share of the visible line on white pixels
MIN_SIDE_SUPPORT = 0.4  # at least one side edge must match this well
MAX_RMS_PX = 2.5  # spread, at working size, of the line pixels around the fitted court lines
CAMERA_HEIGHT_M = (0.3, 8.0)


class AutoCalibrationError(CalibrationError):
    """The court could not be found reliably; click the points instead."""


@dataclass
class AutoCalibration:
    calibration: Calibration
    support: dict[str, float]  # per line: share of its visible length that matched
    rms_px: float  # how far the line pixels sit from the fitted lines, in full-size pixels
    far_error_m: float  # what one rms_px means in depth at the far service line
    camera_height_m: float
    frames_used: int = 1
    used: list[str] = field(default_factory=list)  # lines whose pixels the final fit used

    def summary(self) -> str:
        lines = ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in self.support.items())
        return (
            f"matched: {lines}; spread {self.rms_px:.1f} px, about {self.far_error_m * 100:.0f} cm "
            f"at the far service line; camera {self.camera_height_m:.2f} m high"
        )


# -- public entry points -------------------------------------------------------


def background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of frames from a fixed camera: players vanish, lines stay."""
    if not len(frames):
        raise ValueError("No frames")
    if len(frames) == 1:
        return frames[0]
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def auto_calibrate(frames: list[np.ndarray] | np.ndarray) -> AutoCalibration:
    """Find the court in one frame, or better, several frames from a camera that didn't move."""
    if isinstance(frames, np.ndarray):
        frames = [frames]
    img = background(list(frames))
    h, w = img.shape[:2]
    small, s = _work_size(img)
    found = _find(small)
    S = np.diag([1 / s, 1 / s, 1.0])
    H = S @ found.H  # court -> full-size pixels
    calib = _calibration_from_homography(H, (w, h))
    far = calib.court_to_image(np.array([[2.5, court.FAR_SERVICE_Y], [7.5, court.FAR_SERVICE_Y]]))
    J = calib.jacobian(far)
    rms = found.rms_px / s
    far_m = float(np.mean(np.abs(J[:, 1, :]).max(axis=1))) * rms  # depth error per pixel, times spread
    return AutoCalibration(
        calibration=calib, support=found.support, rms_px=rms, far_error_m=far_m,
        camera_height_m=found.height, frames_used=len(frames), used=found.used,
    )


# -- detection -----------------------------------------------------------------


@dataclass
class _Line:
    l: np.ndarray  # (a, b, c) with a^2 + b^2 = 1, in pixels
    pts: np.ndarray  # the pixels it was fitted to


@dataclass
class _Scene:
    img: np.ndarray
    floor: np.ndarray  # bool: the court floor, holes filled
    white: np.ndarray  # bool: clear painted-line pixels
    faint: np.ndarray  # bool: fainter ones too (far lines seen through the net)
    L: np.ndarray  # float: lightness
    white_near: np.ndarray  # white, a pixel wider, for checking a fit
    faint_near: np.ndarray
    white_px: np.ndarray  # Nx2 (x, y) of the white pixels
    faint_px: np.ndarray
    lines: list[_Line]
    edges: dict[str, np.ndarray]  # left_side / right_side / far_wall -> Nx2 floor-edge pixels
    sides: dict[str, _Line]  # straight side edges found
    far_wall: _Line | None  # straight top edge of the floor, if any


@dataclass
class _Found:
    H: np.ndarray  # court metres -> working-size pixels
    support: dict[str, float]
    rms_px: float
    height: float
    used: list[str]


def _work_size(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = WORK_SIZE / max(h, w)
    if s >= 1:
        return img, 1.0
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return small, small.shape[1] / w


def _find(img: np.ndarray) -> _Found:
    # Usually the bottom of the frame is court; failing that, look further up.
    first = None
    for rows in FLOOR_SAMPLES:
        try:
            sc = _scene(img, rows)
            try:
                return _find_on(sc)
            except AutoCalibrationError as exc:
                if len(sc.sides) < 2:
                    raise
                both = exc
            # A dusty patch or a shadow can fake one side edge; try each side on its own.
            for keep in sc.sides:
                try:
                    return _find_on(replace(sc, sides={keep: sc.sides[keep]}))
                except AutoCalibrationError:
                    pass
            raise both
        except AutoCalibrationError as exc:
            first = first or exc
        except (ValueError, IndexError, np.linalg.LinAlgError, cv2.error) as exc:
            # A degenerate picture (no usable pixels left for some step) is a failed try, not a crash.
            first = first or AutoCalibrationError(
                f"The court finder got stuck on this frame ({type(exc).__name__}). "
                "Try another frame with the slider, or click the points instead."
            )
    raise first


def _find_on(sc: _Scene) -> _Found:
    img = sc.img
    h, w = img.shape[:2]
    if not sc.sides:
        raise AutoCalibrationError(
            "Couldn't find either side edge of the court floor, so the court's width is unknown. "
            "Click the points instead."
        )
    if len(sc.lines) < 2:
        raise AutoCalibrationError(
            "Couldn't find the near service line and the centre line. Try a frame with better light, "
            "or click the points."
        )

    hyps = []
    for N, C in _pairs(sc):
        base = [("near_service_line", N.pts), ("near_center_line", C.pts)]
        base += [(k, v.pts) for k, v in sc.sides.items()]
        # A camera square to the court can't tell how deep the court is from the
        # near lines alone, so also try with the floor's top edge as the far wall.
        tries = [base] + ([base + [("far_wall", sc.far_wall.pts)]] if sc.far_wall is not None else [])
        best = None
        for groups in tries:
            for p0 in _starts(w, h):
                p = _fit_camera(p0, groups, w, h, max_nfev=60)
                s = None if p is None else _score(sc, p)
                if s is not None and (best is None or s[0] > best[0]):
                    best = (s[0], p, base)
        if best is not None:
            hyps.append(best)
    if not hyps:
        raise AutoCalibrationError(
            "None of the lines found fit a padel court seen from behind a back wall. Click the points instead."
        )
    hyps.sort(key=lambda t: -t[0])
    swept = [_sweep_depth(sc, p, groups) for _, p, groups in hyps[:2]]
    p = max(swept, key=lambda q: _score(sc, q)[0] if _score(sc, q) else -np.inf)

    for tol in (6.0, 4.0, 3.0, 2.0, 2.0):
        groups = _assign(sc, p, tol)
        q = _fit_camera(p, groups, w, h)
        p = p if q is None else q
    groups = _assign(sc, p, 2.0)
    used = [n for n, g in groups if len(g)]
    if not used:
        raise AutoCalibrationError(
            "None of the court lines were found where the fit puts them. Click the points instead."
        )
    d = np.concatenate([_dist(p, n, g, w, h) for n, g in groups if len(g)])
    rms = float(np.sqrt(np.mean(np.minimum(d, 4.0) ** 2)))
    _, support = _score(sc, p) or (0.0, {})
    height = float(p[6])

    problems = [
        f"the {k.replace('_', ' ')} only {support.get(k, 0):.0%} matched"
        for k, v in MIN_SUPPORT.items() if support.get(k, 0) < v
    ]
    if max(support.get("left_side", 0), support.get("right_side", 0)) < MIN_SIDE_SUPPORT:
        problems.append("neither side edge matches the floor")
    if max(support.get(k, 0) for k in FAR_CUES) < MIN_FAR_SUPPORT:
        problems.append("nothing at the far end (far service line, far wall or net tape) confirms the fit")
    if rms > MAX_RMS_PX:
        problems.append(f"the line pixels spread {rms:.1f} px around the fit")
    if not CAMERA_HEIGHT_M[0] <= height <= CAMERA_HEIGHT_M[1]:
        problems.append(f"the camera would be {height:.1f} m high")
    if problems:
        raise AutoCalibrationError("The court fit isn't reliable: " + "; ".join(problems) + ". Click the points instead.")
    H = _H(p, w, h)
    return _Found(H=H / abs(H[2, 2]), support=support, rms_px=rms, height=height, used=used)


def _sweep_depth(sc: _Scene, p: np.ndarray, groups) -> np.ndarray:
    """Settle how far away the camera is by what the far end shows.

    With the camera square to the court, the near lines fit a close camera
    with a wide lens as well as a distant one zoomed in; only the far half
    tells them apart. So the camera's distance is stepped through, the rest
    refitted to the near lines each time, and the distance under which the
    net tape, far wall, far service line and far centre line show best wins. For a
    camera at an angle the near lines settle it already, and stepping away
    from that only makes their fit worse, which counts against it.
    """
    h, w = sc.floor.shape

    def value(q):
        if q is None:
            return -np.inf
        s = _score(sc, q)
        if s is None:
            return -np.inf
        d = np.concatenate([_dist(q, n, P, w, h) for n, P in groups if len(P)])
        rms = float(np.sqrt(np.mean(np.minimum(d, 6.0) ** 2)))
        return s[0] - 0.5 * max(0.0, rms - 1.0)

    best = (value(p), p)
    # Distance from the near service line, in even ratios: the view changes as fast
    # between 1 and 2 m away as between 10 and 20 m.
    near = court.NEAR_SERVICE_Y
    for ds in (np.geomspace(1.2, 30.0, 36), None):
        if ds is None:
            d0 = near - float(best[1][5])
            ds = d0 * np.geomspace(0.92, 1.08, 9)
        for y in near - ds:
            q = _fit_camera(best[1], groups, w, h, fix_y=float(y), max_nfev=60)
            v = value(q)
            if v > best[0]:
                best = (v, q)
    q = _fit_camera(best[1], groups, w, h)
    return best[1] if q is None or value(q) < best[0] else q


# Where to read the floor colour, as (top, bottom) fractions of the frame's height,
# middle 40% of its width. Usually the bottom of the frame is court, but a phone
# standing back from the glass sees the surround (often turf) first.
FLOOR_SAMPLES = ((0.72, 0.95), (0.55, 0.72), (0.42, 0.58))


def _scene(img: np.ndarray, rows: tuple[float, float] = FLOOR_SAMPLES[0]) -> _Scene:
    floor = _floor_mask(img, rows)
    white, faint = _white_masks(img, floor)
    lines = _white_lines(white)
    edges = _floor_edges(floor)
    sides = {k: L for k in ("left_side", "right_side") if (L := _fit_edge(edges[k])) is not None}
    # The floor's top outline also runs down the sides; keep only the stretch between them.
    top = edges["far_wall"]
    for L in sides.values():
        top = top[np.abs(top @ L.l[:2] + L.l[2]) > 4.0]
    far_wall = _fit_edge(top, axis=0)
    if far_wall is not None and abs(far_wall.l[0]) > 0.35:  # more than 20 degrees off level: not a far wall
        far_wall = None
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    grow = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # noqa: E731
    px = lambda m: np.column_stack(np.nonzero(m)[::-1]).astype(float)  # noqa: E731
    return _Scene(img, floor, white, faint, L, grow(white), grow(faint), px(white), px(faint),
                  lines, edges, sides, far_wall)


def _floor_mask(img: np.ndarray, rows: tuple[float, float] = FLOOR_SAMPLES[0]) -> np.ndarray:
    """The court floor: pixels the colour of a box in the middle of the frame, joined to it.

    White lines cut the floor into pieces, and the pieces can be lit differently (the near
    end under the camera's own glass is often darker), so the other sample boxes are read
    too and their pieces join in when they are the same colour of floor.
    """
    w = img.shape[1]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ab_ref, floor = _floor_piece(lab, rows)
    if floor is None:
        raise AutoCalibrationError("Couldn't find the court floor at the bottom of the frame. Click the points instead.")
    for other in FLOOR_SAMPLES:
        if other == rows:
            continue
        try:
            ab, piece = _floor_piece(lab, other, min_chroma=3.0)
        except AutoCalibrationError:
            continue
        if piece is None or (piece & ~floor).sum() < w:  # nothing new
            continue
        ratio = np.linalg.norm(ab) / np.linalg.norm(ab_ref)
        same_hue = ab @ ab_ref / (np.linalg.norm(ab) * np.linalg.norm(ab_ref)) > 0.95
        # Only a piece that meets this one along a long border, as across a painted line.
        k = (w // 24) | 1
        near = cv2.dilate(floor.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if same_hue and 0.25 < ratio < 4.0 and (near & piece).sum() > w:
            # The line between the two pieces is court too: whatever has one piece on one
            # side and the other on the opposite side, close by, straight across or up.
            floor = floor | piece | _between(floor, piece, k)
    floor = floor.astype(np.uint8)
    # Fill holes (lines, players, the net) so the floor is one solid region.
    cnts, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(floor, cnts, -1, 1, thickness=cv2.FILLED)
    return floor.astype(bool)


def _between(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Pixels with ``a`` within k pixels on one side and ``b`` on the opposite side."""
    out = np.zeros(a.shape, bool)
    for shape, before, after in (((k, 1), (0, k - 1), (0, 0)), ((1, k), (k - 1, 0), (0, 0))):
        kern = np.ones(shape, np.uint8)
        a0, a1, b0, b1 = (cv2.dilate(m.astype(np.uint8), kern, anchor=anc).astype(bool)
                          for m, anc in ((a, before), (a, after), (b, before), (b, after)))
        out |= (a0 & b1) | (b0 & a1)
    return out


def _floor_piece(lab: np.ndarray, rows: tuple[float, float],
                 min_chroma: float = 6.0) -> tuple[np.ndarray, np.ndarray | None]:
    """The floor colour in one sample box, and the sizeable regions of that colour touching it."""
    h, w = lab.shape[:2]
    box = (slice(int(h * rows[0]), int(h * rows[1])), slice(int(w * 0.3), int(w * 0.7)))
    ref = lab[box].reshape(-1, 3)
    L = ref[:, 0]
    ref = ref[(L >= np.percentile(L, 10)) & (L <= np.percentile(L, 80))]  # skip line paint and shadows
    ab_ref = np.median(ref[:, 1:], axis=0) - 128
    chroma_ref = float(np.linalg.norm(ab_ref))
    if chroma_ref < min_chroma:
        raise AutoCalibrationError(
            "The floor at the bottom of the frame has hardly any colour, so it can't be told from "
            "its surroundings. Click the points instead."
        )
    # Same hue and similar colour strength; brightness is free (the far end is lit differently).
    ab = lab[..., 1:] - 128
    chroma = np.linalg.norm(ab, axis=2)
    cos = (ab @ ab_ref) / (chroma * chroma_ref + 1e-6)
    m = ((cos > 0.93) & (chroma > 0.45 * chroma_ref) & (chroma < 2.2 * chroma_ref)).astype(np.uint8)
    k = max(3, round(w / 160)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, labels = cv2.connectedComponents(m, connectivity=4)
    seed = labels[box].ravel()
    seed = seed[seed > 0]
    if not len(seed):
        return ab_ref, None
    # Every sizeable piece in the sample box: a wide centre line can split the floor in two.
    counts = np.bincount(seed)
    return ab_ref, np.isin(labels, np.flatnonzero(counts >= 0.15 * counts.max()))


def _white_masks(img: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin strokes brighter than the floor and not its colour: clear ones, and faint ones too.

    Lines are usually white, but some courts paint them cream or yellow, so a
    stroke counts if it is paler than the floor or a clearly different hue.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    chroma = np.linalg.norm(lab[..., 1:].astype(np.float32) - 128, axis=2)
    k = max(9, round(w / 70)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se)
    # Compare each stroke with the texture around it, so a faint line in a dim, mesh-covered
    # far end counts as much as a bright one up close. The median ignores the line itself.
    local = cv2.medianBlur(np.clip(tophat, 0, 255).astype(np.uint8), 41).astype(np.float32) + 2.0
    z = tophat / local
    on_floor = cv2.dilate(floor.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ab = lab[..., 1:].astype(np.float32) - 128
    ab_floor = np.median(ab[floor], axis=0) if floor.any() else np.zeros(2)
    cos = (ab @ ab_floor) / (np.linalg.norm(ab, axis=2) * np.linalg.norm(ab_floor) + 1e-6)
    pale = (chroma < 0.85 * cv2.morphologyEx(chroma, cv2.MORPH_CLOSE, se) + 2) | (cos < 0.5)
    white = (z > 4.0) & (tophat > 8) & pale & on_floor
    faint = (z > 2.5) & (tophat > 4) & pale & on_floor
    return white, faint


def _white_lines(white: np.ndarray, max_lines: int = 12) -> list[_Line]:
    """Straight lines through the white pixels, longest first."""
    h, w = white.shape
    segs = cv2.HoughLinesP(white.astype(np.uint8) * 255, 1, np.pi / 720, threshold=max(20, w // 40),
                           minLineLength=max(20, w // 25), maxLineGap=max(4, w // 100))
    if segs is None:
        return []
    ys, xs = np.nonzero(white)
    P = np.column_stack([xs, ys]).astype(np.float64)
    used = np.zeros(len(P), bool)
    out: list[_Line] = []
    for x1, y1, x2, y2 in sorted(segs.reshape(-1, 4).astype(float), key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        l = _line_through((x1, y1), (x2, y2))
        for tol in (3.0, 1.5):
            near = (~used) & (np.abs(P @ l[:2] + l[2]) < tol)
            if near.sum() < 15:
                break
            l = _fit_tls(P[near])
        near = (~used) & (np.abs(P @ l[:2] + l[2]) < 1.5)
        pts = _longest_run(P[near], l, gap=max(6.0, w / 60))
        if len(pts) < max(15, w // 40):
            continue
        l = _fit_tls(pts)
        out.append(_Line(l, pts))
        used |= np.abs(P @ l[:2] + l[2]) < 2.0
        if len(out) >= max_lines:
            break
    return out


def _floor_edges(floor: np.ndarray) -> dict[str, np.ndarray]:
    """Where the floor ends: per row on the left and right, per column at the top.

    Only the widest floor run in each row counts (bridging gaps as narrow as a
    painted line), and ends that reach the frame's border say nothing about the walls.
    """
    h, w = floor.shape
    gap = max(2, w // 50)  # a painted line across the run (the centre line) doesn't end it
    left, right = [], []
    for y in np.flatnonzero(floor.any(axis=1)):
        xs = np.flatnonzero(floor[y])
        breaks = np.flatnonzero(np.diff(xs) > gap)
        starts = np.concatenate([[xs[0]], xs[breaks + 1]])
        ends = np.concatenate([xs[breaks], [xs[-1]]])
        i = int(np.argmax(ends - starts))
        if starts[i] > 1:
            left.append((starts[i] - 0.5, y))
        if ends[i] < w - 2:
            right.append((ends[i] + 0.5, y))
    top = []
    cols = np.flatnonzero(floor.any(axis=0))
    for x in cols:
        y0 = np.flatnonzero(floor[:, x])[0]
        if y0 > 1:
            top.append((x, y0 - 0.5))
    as_arr = lambda a: np.array(a, float).reshape(-1, 2)  # noqa: E731
    return {"left_side": as_arr(left), "right_side": as_arr(right), "far_wall": as_arr(top)}


def _fit_edge(pts: np.ndarray, iters: int = 400, axis: int = 1) -> _Line | None:
    """RANSAC line through floor-edge pixels; None when there is no long straight edge.

    ``axis`` 1 for edges running up the frame (the sides), 0 for across it (the far wall).
    """
    if len(pts) < 30:
        return None
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        if abs(pts[i, axis] - pts[j, axis]) < 10:
            continue
        l = _line_through(pts[i], pts[j])
        n = int((np.abs(pts @ l[:2] + l[2]) < 2.0).sum())
        if n > best_n:
            best, best_n = l, n
    if best is None or best_n < 30:
        return None
    pts_in = _longest_run(pts[np.abs(pts @ best[:2] + best[2]) < 2.0], best, gap=25.0)
    if len(pts_in) < 30:
        return None
    return _Line(_fit_tls(pts_in), pts_in)


# -- the camera ----------------------------------------------------------------


def _H(p: np.ndarray, w: int, h: int) -> np.ndarray:
    """Court -> image homography of a camera (f, rotation vector, position x, y, z)."""
    f, r = p[0], p[1:4]
    R, _ = cv2.Rodrigues(np.asarray(r, float))
    t = -R @ p[4:7]
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def _look_at(pos, target) -> np.ndarray:
    c = np.asarray(pos, float)
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    return cv2.Rodrigues(R)[0].ravel()


def _starts(w: int, h: int):
    """Initial guesses: a camera on or behind the near back wall, looking down the court."""
    for z in (1.3, 3.0):
        for y in (-0.5, -4.0):
            for f in (0.45, 0.9):
                for x in (3.5, 5.0, 6.5):
                    yield np.array([f * max(w, h), *_look_at((x, y, z), (5.0, 10.0, 0.0)), x, y, z])


def _tape_points(p: np.ndarray, w: int, h: int, n: int = 200) -> np.ndarray:
    """The net tape in the image, as in-frame points from post to post."""
    x = np.linspace(0.1, court.WIDTH - 0.1, n)
    z = NET_TAPE_M[0] + NET_TAPE_M[1] * ((x - court.CENTER_X) / court.CENTER_X) ** 2
    P = np.column_stack([x, np.full_like(x, court.NET_Y), z])
    R, _ = cv2.Rodrigues(np.asarray(p[1:4], float))
    c = (P - p[4:7]) @ R.T
    uv = p[0] * c[:, :2] / c[:, 2:3] + [w / 2, h / 2]
    ok = (c[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    return uv[ok]


def _dist(p: np.ndarray, name: str, P: np.ndarray, w: int, h: int, Hinv_T: np.ndarray | None = None) -> np.ndarray:
    """Signed pixel distance of P from court line ``name`` as the camera p sees it.

    ``Hinv_T``, the inverse transpose of the camera's homography, saves recomputing it.
    """
    if name == "net_tape":
        T = _tape_points(p, w, h, 400)
        if len(T) < 2:
            return np.full(len(P), 50.0)
        i = np.argmin(((P[:, None, :] - T[None, :, :]) ** 2).sum(axis=2), axis=1)
        j = np.clip(i, 1, len(T) - 1)
        d = T[j] - T[j - 1]
        nrm = np.column_stack([-d[:, 1], d[:, 0]]) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        return ((P - T[i]) * nrm).sum(axis=1)
    li = (np.linalg.inv(_H(p, w, h)).T if Hinv_T is None else Hinv_T) @ MODEL_LINES[name]
    li = li / np.linalg.norm(li[:2])
    return P @ li[:2] + li[2]


def _fit_camera(p0: np.ndarray, groups, w: int, h: int, fix_y: float | None = None,
                max_nfev: int = 200) -> np.ndarray | None:
    """Least squares: the camera under which each group's pixels sit on its court line.

    ``fix_y`` holds the camera's distance along the court at that value.
    """
    groups = [(n, P) for n, P in groups if len(P) >= 5]
    if len(groups) < 3:
        return None
    # A few hundred pixels per line pin it down as well as thousands, much faster.
    groups = [(n, P[:: max(1, len(P) // 300)]) for n, P in groups]
    p0 = np.asarray(p0, float).copy()
    free = list(range(7))
    if fix_y is not None:
        p0[5] = fix_y
        free = [0, 1, 2, 3, 4, 6]
    # Each line counts about the same however many pixels it has, so thick near
    # paint doesn't drown out the thin far lines.
    wts = [1.0 / np.sqrt(len(P)) for _, P in groups]

    def resid(x):
        p = p0.copy()
        p[free] = x
        try:
            Hit = np.linalg.inv(_H(p, w, h)).T
            return np.concatenate([wt * _dist(p, n, P, w, h, Hit) for (n, P), wt in zip(groups, wts)])
        except np.linalg.LinAlgError:
            return np.full(sum(len(P) for _, P in groups), 1e3)

    lo = np.array([0.15 * max(w, h), -np.inf, -np.inf, -np.inf, -30.0, -40.0, 0.05])[free]
    hi = np.array([10.0 * max(w, h), np.inf, np.inf, np.inf, 40.0, 60.0, 30.0])[free]
    try:
        sol = least_squares(resid, np.clip(p0[free], lo + 1e-6, hi - 1e-6), bounds=(lo, hi),
                            loss="soft_l1", f_scale=1.0, max_nfev=max_nfev)
    except (ValueError, np.linalg.LinAlgError):
        return None
    out = p0.copy()
    out[free] = sol.x
    return out


# -- scoring and assignment ------------------------------------------------------


def _project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def _samples(H: np.ndarray, name: str, shape, step_px: float = 2.0, offset: float = 0.0) -> np.ndarray:
    """In-frame pixels along court line ``name`` (shifted ``offset`` metres sideways), in front of the camera."""
    h, w = shape
    (x0, y0), (x1, y1) = EXTENT[name]
    a, b = np.array([x0, y0]), np.array([x1, y1])
    normal = np.array([y1 - y0, x0 - x1]) / np.hypot(x1 - x0, y1 - y0)
    ends, _ = _project(H, np.array([a, b]))
    n = int(np.clip(np.linalg.norm(ends[1] - ends[0]) / step_px, 20, 3000))
    pts = np.linspace(a, b, n) + offset * normal
    img, depth = _project(H, pts)
    ok = (depth > 0) & (img[:, 0] >= 0) & (img[:, 0] < w - 1) & (img[:, 1] >= 0) & (img[:, 1] < h - 1)
    return img[ok]


def _score(sc: _Scene, p: np.ndarray) -> tuple[float, dict[str, float]] | None:
    """How well the court model seen by camera p lands on the lines, floor edges and net tape."""
    h, w = sc.floor.shape
    H = _H(p, w, h)
    _, depth = _project(H, np.array([[0, 10], [10, 10], [0, 20], [10, 20]], float))
    if not (depth > 0).all():
        return None  # the net and the far end are always in front of the camera
    (fl, fr, nl, nr), _ = _project(H, np.array([[0, 20], [10, 20], [0, 10], [10, 10]], float))
    if not (fl[0] < fr[0] and nl[0] < nr[0] and fl[1] < nl[1] and fr[1] < nr[1]):
        return None  # seen from behind the near wall, left is left and the far end higher up
    support: dict[str, float] = {}
    for name in PAINTED:
        q = _samples(H, name, (h, w))
        if len(q) < 10:
            continue
        if name == "far_service_line":
            support[name] = _ridge_share(sc, q)
        else:
            u = q.round().astype(int)
            mask = sc.white_near if name.startswith("near") else sc.faint_near
            support[name] = float(mask[u[:, 1], u[:, 0]].mean())
    if "near_service_line" not in support:
        return None
    for name in EDGES:
        on = _samples(H, name, (h, w))
        if len(on) < 10:
            continue
        # Floor 3 px inside the edge and none 3 px outside it.
        inward, _ = _project(H, np.array([[5.0, 10.0]]))
        d = on[-1] - on[0]
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        if (inward[0] - on.mean(axis=0)) @ nrm < 0:
            nrm = -nrm
        a = np.clip((on + 3 * nrm).round().astype(int), 0, [w - 1, h - 1])
        b = np.clip((on - 3 * nrm).round().astype(int), 0, [w - 1, h - 1])
        support[name] = float((sc.floor[a[:, 1], a[:, 0]] & ~sc.floor[b[:, 1], b[:, 0]]).mean())
    tape = _tape_points(p, w, h)
    if len(tape) >= 20:
        support["net_tape"] = _ridge_share(sc, tape)
    # White just beside the near service line means a different line was fitted.
    false = 0.0
    for off in (-0.5, 0.5):
        u = _samples(H, "near_service_line", (h, w), offset=off).round().astype(int)
        if len(u):
            false += float(sc.white_near[u[:, 1], u[:, 0]].mean())
    weight = {"near_service_line": 2.0, "near_center_line": 2.0, "net_tape": 1.5}
    score = sum(weight.get(k, 1.0) * v for k, v in support.items()) - 2 * false
    return score, support


def _snap(sc: _Scene, H: np.ndarray, name: str) -> np.ndarray:
    """Points along the faint painted line closest to where H puts court line ``name``.

    Far lines are seen through the net, too faint and broken for the white
    mask, but averaged along their whole length they still show as a bright
    ridge across the line. Shifts and small turns of the predicted line are
    tried, within about a metre of court, and the one with the strongest
    ridge wins if the ridge is clear. Returns no points when none is.
    """
    h, w = sc.L.shape
    pts = _samples(H, name, (h, w), step_px=1.0)
    if len(pts) < 30:
        return np.zeros((0, 2))
    d = pts[-1] - pts[0]
    d = d / np.linalg.norm(d)
    mid = pts.mean(axis=0)
    t = (pts - mid) @ d
    # How many pixels one metre of depth is here: the search range.
    (x0, y0), (x1, y1) = EXTENT[name]
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    a, _ = _project(H, np.array([c - [0, 0.75], c + [0, 0.75]]))
    reach = float(np.clip(abs((a[1] - a[0]) @ np.array([-d[1], d[0]])), 2.0, 25.0))
    k = 3.0  # ridge test: brighter than the floor this far to either side
    step = int(round(k / 0.25))
    offs = np.arange(-reach - k, reach + k + 0.01, 0.25)
    inner = (np.abs(offs) <= reach)[step:-step]
    best = (0.0, None, None)
    for ang in np.deg2rad(np.arange(-1.0, 1.01, 0.25)):
        dd = np.array([np.cos(ang) * d[0] - np.sin(ang) * d[1], np.sin(ang) * d[0] + np.cos(ang) * d[1]])
        nn = np.array([-dd[1], dd[0]])
        base = mid + t[:, None] * dd
        prof = np.array([_values_at(sc.L, base + o * nn).mean() for o in offs])
        ridge = prof[step:-step] - np.maximum(prof[:-2 * step], prof[2 * step:])
        ridge = np.where(inner, ridge, -np.inf)
        j = int(np.argmax(ridge))
        if ridge[j] > best[0]:
            best = (ridge[j], base + offs[step + j] * nn, nn)
    if best[1] is None or best[0] < 3.0:
        return np.zeros((0, 2))
    q, nn = best[1], best[2]
    # Keep the stretches that actually show the line.
    v = _values_at(sc.L, q) - np.maximum(_values_at(sc.L, q + k * nn), _values_at(sc.L, q - k * nn))
    q = q[cv2.blur(v.reshape(-1, 1).astype(np.float32), (1, 15)).ravel() > 2.0]
    return q[::2] if len(q) >= 30 else np.zeros((0, 2))


def _ridge_share(sc: _Scene, q: np.ndarray, k: float = 3.0) -> float:
    """Share of the points q (along one line or curve) that sit on a bright ridge across it."""
    d = np.gradient(q, axis=0)
    nn = np.column_stack([-d[:, 1], d[:, 0]]) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    best = 0.0
    for o in (-1.0, 0.0, 1.0):  # a pixel's slack
        p = q + o * nn
        v = _values_at(sc.L, p) - np.maximum(_values_at(sc.L, p + k * nn), _values_at(sc.L, p - k * nn))
        best = max(best, float((cv2.blur(v.reshape(-1, 1).astype(np.float32), (1, 9)).ravel() > 2.0).mean()))
    return best


def _values_at(img: np.ndarray, q: np.ndarray) -> np.ndarray:
    return cv2.remap(img, q[:, 0].astype(np.float32).reshape(1, -1), q[:, 1].astype(np.float32).reshape(1, -1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()


def _assign(sc: _Scene, p: np.ndarray, tol: float):
    """Pixels within ``tol`` of each line as camera p sees it, inside the line's extent."""
    h, w = sc.L.shape
    H = _H(p, w, h)
    Hinv = np.linalg.inv(H)
    out = []
    for name in MODEL_LINES:
        if name == "far_service_line":
            out.append((name, _snap(sc, H, name)))
            continue
        if name in EDGES:
            P = sc.edges[name]
        else:
            P = sc.white_px if name.startswith("near") else sc.faint_px
        if not len(P):
            out.append((name, P))
            continue
        P = P[np.abs(_dist(p, name, P, w, h)) < tol]
        if len(P):
            c, _ = _project(Hinv, P)
            (x0, y0), (x1, y1) = EXTENT[name]
            m = 0.25
            inside = (c[:, 0] > min(x0, x1) - m) & (c[:, 0] < max(x0, x1) + m)
            inside &= (c[:, 1] > min(y0, y1) - m) & (c[:, 1] < max(y0, y1) + m)
            P = P[inside]
        # A far line must show along a fair stretch, or it's noise.
        if name not in ("near_service_line", "near_center_line") and len(P) < 15:
            P = P[:0]
        out.append((name, P))
    # The tape: white pixels close to where the camera puts it.
    T = _tape_points(p, w, h)
    P = sc.white_px
    if len(T) >= 20 and len(P):
        lo, hi = T.min(axis=0) - tol, T.max(axis=0) + tol
        P = P[((P >= lo) & (P <= hi)).all(axis=1)]
        P = P[np.abs(_dist(p, "net_tape", P, w, h)) < tol] if len(P) else P
        out.append(("net_tape", P if len(P) >= 15 else P[:0]))
    return out


def _calibration_from_homography(H: np.ndarray, size: tuple[int, int]) -> Calibration:
    """A normal, point-based calibration: every court keypoint the fit puts inside the frame."""
    w, h = size
    names = list(court.KEYPOINTS)
    img, depth = _project(H, np.array([court.KEYPOINTS[n] for n in names]))
    m = 0.02 * max(w, h)
    pts = {
        n: (round(float(u), 1), round(float(v), 1))
        for n, (u, v), d in zip(names, img, depth)
        if d > 0 and -m <= u <= w + m and -m <= v <= h + m
    }
    try:
        return Calibration.from_points(pts, (w, h))
    except CalibrationError as exc:
        raise AutoCalibrationError(f"The court was found, but too little of it is in the frame: {exc}") from exc


def _pairs(sc: _Scene):
    """(near service line, centre line) candidates: a flatter line crossed by a steeper one."""
    h, w = sc.floor.shape
    for N, C in permutations(sc.lines, 2):
        # The centre line runs up the frame more steeply than the service line.
        if abs(C.l[0]) <= abs(N.l[0]):
            continue
        p = _meet(N.l, C.l)
        if p is None or not (-0.1 * w < p[0] < 1.1 * w and 0 < p[1] < h):
            continue
        # The centre line's paint is mostly above (further than) the service line.
        tN = N.pts @ np.array([-N.l[1], N.l[0]])
        if not (tN.min() - 0.05 * w <= p @ np.array([-N.l[1], N.l[0]]) <= tN.max() + 0.05 * w):
            continue
        if np.mean(C.pts[:, 1] < p[1] + 3) < 0.6:
            continue
        yield N, C


# -- small geometry helpers ------------------------------------------------------


def _line_through(p, q) -> np.ndarray:
    l = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    return l / np.linalg.norm(l[:2])


def _fit_tls(P: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = cv2.fitLine(P.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    l = np.array([vy, -vx, vx * y0 - vy * x0], float)
    return l / np.linalg.norm(l[:2])


def _longest_run(P: np.ndarray, l: np.ndarray, gap: float) -> np.ndarray:
    """The pixels of the longest stretch along l with no gap wider than ``gap``."""
    if len(P) == 0:
        return P
    t = P @ np.array([-l[1], l[0]])
    o = np.argsort(t)
    t, P = t[o], P[o]
    cuts = np.flatnonzero(np.diff(t) > gap)
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts + 1, [len(t)]])
    i = int(np.argmax(ends - starts))
    return P[starts[i]:ends[i]]


def _meet(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def draw_detection(img: np.ndarray) -> np.ndarray:
    """Debug picture at working size: floor tinted green, white pixels red, faint ones orange,
    fitted white lines yellow and numbered, side edges magenta."""
    small, _ = _work_size(img)
    sc = _scene(small)
    out = small.copy()
    out[sc.floor] = (0.6 * out[sc.floor] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
    out[sc.faint] = (0, 140, 255)
    out[sc.white] = (0, 0, 255)
    for i, L in enumerate(sc.lines):
        t = L.pts @ np.array([-L.l[1], L.l[0]])
        a, b = L.pts[t.argmin()], L.pts[t.argmax()]
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 255, 255), 1)
        cv2.putText(out, str(i), tuple(((a + b) / 2).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for L in sc.sides.values():
        cv2.line(out, tuple(L.pts[0].astype(int)), tuple(L.pts[-1].astype(int)), (255, 0, 255), 2)
    return out
