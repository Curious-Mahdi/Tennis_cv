"""Tests for analytics.ball_track and analytics.movement (no model weights needed)."""
import math

from analytics import clean_ball_track, cumulative_distance, heatmap, player_movement
from analytics.ball_track import DETECTED, INTERPOLATED, REJECTED, UNAVAILABLE


def line(n, x0=100.0, y0=100.0, vx=10.0, vy=5.0):
    return [(x0 + vx * i, y0 + vy * i) for i in range(n)]


def test_isolated_false_positive_is_rejected_and_not_interpolated_through():
    raw = line(10)
    raw[5] = (1100.0, 600.0)          # far from the track
    raw[4] = None
    t = clean_ball_track(raw)
    assert t.status[5] == REJECTED
    assert t.xy[5] is None
    # 3 -> 6 is a 3-frame gap inside one tracklet, so 4 and 5 are interpolated ON the line
    assert t.status[4] == INTERPOLATED
    assert abs(t.xy[4][0] - 140.0) < 1e-9


def test_long_gap_is_not_interpolated():
    raw = line(20)
    for f in range(5, 15):
        raw[f] = None
    t = clean_ball_track(raw)
    assert all(t.status[f] == UNAVAILABLE for f in range(5, 15))


def test_no_interpolation_across_event():
    raw = line(10)
    raw[4] = raw[5] = None
    t = clean_ball_track(raw, event_frames=[4])
    assert t.status[4] == UNAVAILABLE and t.status[5] == UNAVAILABLE


def test_direction_change_is_preserved_exactly():
    down = [(500.0, 100.0 + 20 * i) for i in range(6)]
    up = [(500.0 + 10 * i, 200.0 - 15 * i) for i in range(1, 6)]
    raw = down + up
    t = clean_ball_track(raw)
    assert all(t.status[i] == DETECTED for i in range(len(raw)))
    assert t.xy == raw


def test_short_tracklet_rejected():
    raw = [None] * 10
    raw[3] = (10.0, 10.0)
    t = clean_ball_track(raw)
    assert t.counts()["rejected"] == 1 and t.counts()["retained"] == 0


def test_cumulative_distance_does_not_bridge_gaps():
    series = [(0, (0.0, 0.0)), (1, (1.0, 0.0)), (5, (100.0, 0.0)), (6, (101.0, 0.0))]
    cum = cumulative_distance(series, 7)
    assert cum[0] == 0.0 and cum[1] == 1.0 and cum[6] == 2.0


def test_heatmap_shares_sum_to_one():
    h = heatmap([(1.0, 1.0), (1.2, 1.1), (5.0, 20.0)])
    assert math.isclose(sum(map(sum, h["share"])), 1.0, abs_tol=1e-3)


def test_player_movement_uses_only_measured_frames():
    frames = []
    for f in range(4):
        court = {"status": "measured", "xy_m": [1.0 + f, 23.0], "xy_m_smoothed": [1.0 + f, 23.0]}
        if f == 2:
            court = {"status": "outside_plausible_area", "xy_m": None}
        frames.append({"f": f, "players": {"1": {"status": "detected", "court": court}}})
    m = player_movement(frames, "1", "near", 30.0)
    assert m["frames_measured"] == 3
    assert m["path_m_smoothed"] == 1.0           # 0->1 only; 1->3 is a gap
    assert m["gaps_not_bridged"] == 1
    assert m["series"]["depth_m"][0] == round(23.76 - 23.0, 3)
