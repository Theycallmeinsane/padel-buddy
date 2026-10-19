import numpy as np
import pytest

from padel_engine import court
from padel_engine.calibration import Calibration, CalibrationError
from synthetic import H, W, Camera

FAR_AND_SERVICE = [
    "far_left_corner", "far_right_corner",
    "far_service_left", "far_service_right",
    "near_service_left", "near_service_right",
]


def clicked(cam: Camera, names, noise_px: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts = cam.ground([court.KEYPOINTS[n] for n in names])
    pts = pts + rng.normal(0, noise_px, pts.shape) if noise_px else pts
    return {n: tuple(p) for n, p in zip(names, pts)}


def test_exact_points_recover_court_coordinates():
    cam = Camera()
    calib = Calibration.from_points(clicked(cam, FAR_AND_SERVICE), (W, H))
    probe = np.array([[1.0, 2.0], [5.0, 10.0], [8.5, 18.0], [9.9, 0.1]])
    np.testing.assert_allclose(calib.image_to_court(cam.ground(probe)), probe, atol=1e-6)
    assert calib.reprojection_error_m < 1e-4


def test_noisy_clicks_still_accurate_to_a_few_centimetres():
    cam = Camera()
    calib = Calibration.from_points(clicked(cam, list(court.KEYPOINTS)[:11], noise_px=2.0), (W, H))
    probe = np.array([[2.0, 5.0], [7.5, 15.0], [5.0, 8.0]])
    err = np.linalg.norm(calib.image_to_court(cam.ground(probe)) - probe, axis=1)
    assert err.max() < 0.25


def test_round_trip_and_resolution_scaling(tmp_path):
    cam = Camera()
    calib = Calibration.from_points(clicked(cam, FAR_AND_SERVICE), (W, H))
    calib.save(tmp_path / "c.json")
    loaded = Calibration.load(tmp_path / "c.json")
    np.testing.assert_allclose(loaded.H, calib.H, atol=1e-9)

    big = loaded.scaled_to((W * 2, H * 2))
    px = cam.ground([(3.0, 12.0)]) * 2
    np.testing.assert_allclose(big.image_to_court(px), [[3.0, 12.0]], atol=1e-6)


def test_court_to_image_inverts_image_to_court():
    cam = Camera()
    calib = Calibration.from_points(clicked(cam, FAR_AND_SERVICE), (W, H))
    pts = np.array([[0.5, 0.5], [9.0, 19.0]])
    np.testing.assert_allclose(calib.image_to_court(calib.court_to_image(pts)), pts, atol=1e-6)


def test_rejects_too_few_or_collinear_points():
    cam = Camera()
    with pytest.raises(CalibrationError):
        Calibration.from_points(clicked(cam, FAR_AND_SERVICE[:3]), (W, H))
    line = ["far_service_left", "far_service_center", "far_service_right", "far_left_corner"]
    with pytest.raises(CalibrationError):
        Calibration.from_points(clicked(cam, line), (W, H))
    with pytest.raises(CalibrationError):
        Calibration.from_points({"not_a_point": (1, 2), **clicked(cam, FAR_AND_SERVICE)}, (W, H))


def test_jacobian_matches_finite_differences():
    cam = Camera()
    calib = Calibration.from_points(clicked(cam, FAR_AND_SERVICE), (W, H))
    px = cam.ground([(2.0, 3.0), (7.0, 18.0)])
    J = calib.jacobian(px)
    eps = 1e-3
    for d in range(2):
        step = np.zeros(2)
        step[d] = eps
        num = (calib.image_to_court(px + step) - calib.image_to_court(px - step)) / (2 * eps)
        np.testing.assert_allclose(J[:, :, d], num, rtol=1e-4, atol=1e-8)
    # One pixel moves a far-wall point much further than a near one.
    assert np.linalg.norm(J[1]) > 3 * np.linalg.norm(J[0])


def test_lines_behind_the_camera_are_not_drawn():
    """A phone on the back glass can have the near wall under or behind it."""
    from padel_engine.render import court_polylines

    cam = Camera(pos=(5.0, 2.0, 2.0), look_at=(5.0, 11.0, 0.0))
    calib = Calibration.from_points(clicked(cam, FAR_AND_SERVICE), (W, H))
    assert calib.in_front([(5.0, 10.0), (5.0, 20.0)]).all()
    assert not calib.in_front([(5.0, 0.0)]).any()

    lines = dict(court_polylines(calib))
    assert "near_wall" not in lines and "far_wall" in lines
    # Side walls stop where they pass behind the camera, and what is left is exact.
    left = lines["left_wall"]
    assert len(left) < 50
    np.testing.assert_allclose(calib.image_to_court(left)[:, 0], 0.0, atol=1e-6)
