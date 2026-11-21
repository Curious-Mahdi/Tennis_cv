"""
Tests for analytics.ball_flight against a synthetic camera and a synthetic rally with known
ground truth: does the method recover the truth, and does it refuse when it should?
"""
import numpy as np
import pytest

from analytics.ball_flight import (COURT_KP_M, Camera, _flight, _gap_frames, cameras,
                                   estimate_flights, observed_bounce)

W, H, F, FPS = 1280, 720, 1500.0, 30.0


def _true_camera():
    """Broadcast-like camera 12 m behind and 9 m above the near baseline."""
    C = np.array([5.485, 23.76 + 12.0, -9.0])          # court z axis points down
    T = np.array([5.485, 11.88, 0.0])
    z = (T - C) / np.linalg.norm(T - C)
    down = np.array([0.0, 0.0, 1.0])
    x = np.cross(down, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    return Camera(K, rvec, (-R @ C).reshape(3, 1), -1.0, 0.0)


@pytest.fixture(scope="module")
def rig():
    true = _true_camera()
    kps = [true.project([(x, y, 0) for x, y in COURT_KP_M]).ravel() for _ in range(80)]
    cams, focal = cameras(kps, W, H)
    return true, cams, focal


def test_camera_recovered_from_court_keypoints(rig):
    true, cams, focal = rig
    assert abs(focal - F) / F < 0.03
    # a point 2 m above the court projects where the true camera puts it
    p_true = true.project([(3.0, 10.0, 2.0)])[0]
    p_est = cams[0].project([(3.0, 10.0, 2.0)])[0]
    assert np.hypot(*(p_true - p_est)) < 3.0
    # height moves a point UP the image
    a, b = cams[0].project([(5, 12, 0), (5, 12, 1)])
    assert b[1] < a[1]


def test_ray_to_height_inverts_projection(rig):
    _, cams, _ = rig
    u, v = cams[0].project([(2.0, 20.0, 1.3)])[0]
    x, y = cams[0].ray_to_height(u, v, 1.3)
    assert abs(x - 2.0) < 1e-6 and abs(y - 20.0) < 1e-6


def test_synthetic_rally_is_recovered_and_labelled(rig):
    true, cams, _ = rig
    # contact (near player, 1.0 m) -> unseen bounce -> contact (far player, 1.2 m)
    c1 = (0, 2.4, 23.0, 1.0)
    b1 = (30, 6.0, 6.0, 0.0)
    c2 = (50, 7.0, 1.6, 1.2)
    det = {}
    for f in range(0, 51):
        if 27 <= f <= 33:            # ball not detected through the bounce
            continue
        a, b = (c1, b1) if f <= 30 else (b1, c2)
        det[f] = tuple(true.project([_flight(f, a, b, FPS)])[0])
    events = [{"frame": 0, "type": "contact", "hitter": 1}, {"frame": 30, "type": "bounce"},
              {"frame": 50, "type": "contact", "hitter": 2}]
    out = estimate_flights(events, det, cams, FPS, 60, hitter_xy={0: (2.0, 23.6), 50: (7.3, 0.6)})
    bounce = out["events"][1]
    assert bounce["status"] == "estimated"
    err = np.hypot(bounce["xy_m"][0] - 6.0, bounce["xy_m"][1] - 6.0)
    assert err <= max(0.5, bounce["uncertainty_m"])
    assert all(s["status"] in ("shown", "shown_uncertain") for s in out["segments"])
    assert all(s["median_residual_px"] < 2.0 for s in out["segments"])
    # nothing is drawn outside the modelled span
    assert max(out["frames"]) <= 50


def test_contradicted_flight_is_withheld(rig):
    true, cams, _ = rig
    a, b = (0, 2.4, 23.0, 1.0), (30, 6.0, 6.0, 0.0)
    # Half-way through, the detections jump 150 px sideways and stay there - what a missed
    # hit looks like. (A constant offset would not do: height and depth can absorb it.)
    det = {f: tuple(true.project([_flight(f, a, b, FPS)])[0] + np.array([150.0 if f > 12 else 0.0, 0.0]))
           for f in range(0, 26)}
    events = [{"frame": 0, "type": "contact", "hitter": 1}, {"frame": 30, "type": "bounce"}]
    out = estimate_flights(events, det, cams, FPS, 40, hitter_xy={0: (2.0, 23.6)})
    assert out["segments"][0]["status"] == "withheld"
    assert not out["frames"]


def test_observed_bounce_needs_a_visible_impact():
    falling = {f: (500.0, 100.0 + 25 * f) for f in range(0, 11)}
    assert observed_bounce(6, falling) is None                     # no kink, no bounce
    kink = {**falling, **{f: (500.0 + 5 * (f - 6), 250.0 + 3 * (f - 6)) for f in range(7, 11)}}
    assert observed_bounce(6, kink)[0] == 6


def test_unseen_bounce_is_searched_inside_the_detection_gap():
    det = {f: (0.0, 0.0) for f in list(range(0, 20)) + list(range(25, 40))}
    assert _gap_frames(22, det, 40) == [20, 21, 22, 23, 24]
