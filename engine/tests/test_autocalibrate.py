import cv2
import numpy as np
import pytest

from padel_engine import court
from padel_engine.autocalibrate import NET_TAPE_M, AutoCalibrationError, auto_calibrate
from synthetic import H, W, Camera

FLOOR = (170, 110, 40)  # BGR: court blue
SURROUND = (60, 120, 50)  # turf green beyond the side walls
PAINT = (235, 240, 240)


def court_frame(cam: Camera, noise: float = 4.0, seed: int = 0, players: bool = True,
                near_floor: tuple[int, int, int] | None = None) -> np.ndarray:
    """A camera's view of an empty court: blue floor, white lines 5 cm wide, net with its tape.

    ``near_floor`` paints the floor behind the near service line another shade, as when
    the camera's own glass or a roof edge darkens it.
    """
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), SURROUND, np.uint8)
    img[: H // 4] = (25, 25, 30)  # night sky
    _fill_ground(img, cam, [(0, 0.0), (10, 0.0), (10, 20), (0, 20)], FLOOR)
    if near_floor is not None:
        _fill_ground(img, cam, [(0, 0.0), (10, 0.0), (10, court.NEAR_SERVICE_Y), (0, court.NEAR_SERVICE_Y)], near_floor)
    for name, ((x1, y1), (x2, y2)) in court.LINES.items():
        if name.endswith("wall") or name == "net":
            continue
        _paint_ground_line(img, cam, (x1, y1), (x2, y2), 0.05)
    # The net: a translucent dark band from the floor to the tape, and the tape.
    xs = np.linspace(0, 10, 60)
    top = NET_TAPE_M[0] + 0.025 + NET_TAPE_M[1] * ((xs - 5) / 5) ** 2
    up = cam.project(np.column_stack([xs, np.full_like(xs, 10.0), top]))
    down = cam.project(np.column_stack([xs, np.full_like(xs, 10.0), np.zeros_like(xs)]))
    band = np.concatenate([up, down[::-1]]).round().astype(np.int32)
    net = img.copy()
    cv2.fillPoly(net, [band], (20, 20, 20))
    img = cv2.addWeighted(img, 0.65, net, 0.35, 0)
    tape_lo = cam.project(np.column_stack([xs, np.full_like(xs, 10.0), top - 0.05]))
    tape = np.concatenate([up, tape_lo[::-1]]).round().astype(np.int32)
    cv2.fillPoly(img, [tape], PAINT)
    if players:
        # Players stand somewhere else in every frame, as in a rally.
        for x, y in rng.uniform([1, 1], [9, 19], (4, 2)):
            foot = cam.ground([(x, y)])[0]
            head = cam.project([(x, y, 1.8)])[0]
            hgt = foot[1] - head[1]
            cv2.rectangle(img, (int(foot[0] - hgt / 6), int(head[1])), (int(foot[0] + hgt / 6), int(foot[1])),
                          (40, 40, 200), -1)
    img = img.astype(np.float32) + rng.normal(0, noise, img.shape)
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)


def _fill_ground(img, cam, poly, color, n=40):
    pts = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        pts.append(np.linspace(a, b, n))
    px = cam.ground(np.concatenate(pts)).round().astype(np.int32)
    cv2.fillPoly(img, [px], color)


def _paint_ground_line(img, cam, a, b, width):
    a, b = np.array(a, float), np.array(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]]) * width / 2
    _fill_ground(img, cam, [tuple(a - n), tuple(b - n), tuple(b + n), tuple(a + n)], PAINT, n=80)


def court_error_m(calib, cam: Camera) -> np.ndarray:
    grid = np.array([(x, y) for x in np.linspace(0.5, 9.5, 7) for y in np.linspace(1.0, 19.5, 12)])
    px = cam.ground(grid)
    inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    return np.linalg.norm(calib.image_to_court(px[inside]) - grid[inside], axis=1)


CAMERAS = {
    # A phone on a stand at head height, a little off-centre and tilted.
    "off_centre": Camera(pos=(3.6, -0.4, 1.35), look_at=(5.3, 11.0, 0.0), f=640.0, roll_deg=1.0),
    # Square to the court in the middle: the near lines alone can't tell how far away it is.
    "square_on": Camera(pos=(5.0, -0.1, 1.65), look_at=(5.0, 12.0, 0.0), f=600.0, roll_deg=0.0),
    # Mounted high behind the glass.
    "high": Camera(pos=(5.4, -2.5, 3.2), look_at=(5.0, 11.0, 0.0), f=820.0, roll_deg=1.5),
}


@pytest.mark.parametrize("name", list(CAMERAS))
def test_finds_the_court_from_its_lines(name):
    cam = CAMERAS[name]
    frames = [court_frame(cam, seed=s) for s in range(3)]
    found = auto_calibrate(frames)
    err = court_error_m(found.calibration, cam)
    # At 1280x720 a pixel at the far wall is 15-30 cm of depth, so that's the scale of error to expect there.
    assert np.median(err) < 0.15, (name, found.summary())
    assert err.max() < 0.40, (name, found.summary())
    assert abs(found.camera_height_m - _height(cam)) < 0.15
    # The far corners come out of the fit, not clicks.
    assert {"far_left_corner", "far_right_corner"} <= set(found.calibration.points)


def _height(cam: Camera) -> float:
    return float(-(cam.R.T @ cam.t)[2])


def test_near_end_in_another_shade_with_the_side_lines_out_of_frame():
    # As at Matchpoint: a phone close to the glass sees the near service line run off both
    # edges of the frame, and the floor behind it is darker and greyer than the rest.
    cam = Camera(pos=(5.0, -0.13, 1.7), look_at=(5.0, 6.1, 0.0), f=640.0, roll_deg=0.5)
    ends = cam.ground([(0.0, court.NEAR_SERVICE_Y), (10.0, court.NEAR_SERVICE_Y)])
    assert ends[0, 0] < 0 and ends[1, 0] > W  # the near sides really are out of frame
    frames = [court_frame(cam, seed=s, near_floor=(105, 80, 55)) for s in range(3)]
    found = auto_calibrate(frames)
    err = court_error_m(found.calibration, cam)
    assert np.median(err) < 0.15, found.summary()
    assert err.max() < 0.40, found.summary()


def test_players_on_the_lines_are_removed_by_combining_frames():
    cam = CAMERAS["off_centre"]
    found = auto_calibrate([court_frame(cam, seed=s) for s in range(5)])
    assert np.median(court_error_m(found.calibration, cam)) < 0.10


def test_says_why_when_there_is_no_court():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    with pytest.raises(AutoCalibrationError):
        auto_calibrate(noise)
    blank = np.full((H, W, 3), FLOOR, np.uint8)  # a floor but no lines
    with pytest.raises(AutoCalibrationError, match="(?i)click the points"):
        auto_calibrate(blank)


def test_result_is_an_ordinary_point_calibration(tmp_path):
    cam = CAMERAS["off_centre"]
    found = auto_calibrate([court_frame(cam, seed=s) for s in range(3)])
    found.calibration.save(tmp_path / "c.json")
    from padel_engine.calibration import Calibration

    loaded = Calibration.load(tmp_path / "c.json")
    np.testing.assert_allclose(loaded.image_to_court([[640, 500]]), found.calibration.image_to_court([[640, 500]]),
                               atol=1e-6)
