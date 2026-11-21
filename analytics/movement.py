"""
analytics/movement.py
─────────────────────
Player movement measures from per-frame court positions (metres, court frame of
scripts/export_web_data.py: x across 0..10.97, y along 0 = far baseline .. 23.76 = near).

Only frames where a player was actually measured are used. A missing frame breaks the
path: distance is never accumulated across a gap, because the route taken while the
player was not tracked is unknown.

Noise matters here. Feet are the bottom of a detector box, and at the far baseline one
image pixel spans roughly 0.1 m in depth, so frame-to-frame jitter adds path length that
was never run. Both the raw sum and the sum over the pipeline's Kalman-smoothed positions
are reported; the gap between them is a direct measure of how much is jitter.
"""
from __future__ import annotations

import math

COURT_W, COURT_L = 10.97, 23.76


def _steps(series: list[tuple[int, tuple[float, float]]]) -> list[tuple[int, float]]:
    out = []
    for (f0, p0), (f1, p1) in zip(series, series[1:]):
        if f1 - f0 == 1:
            out.append((f1, math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    return out


def cumulative_distance(series: list[tuple[int, tuple[float, float]]], n_frames: int) -> list[float | None]:
    """Running path length per frame (None before the first measurement)."""
    steps = dict(_steps(series))
    measured = {f for f, _ in series}
    out: list[float | None] = [None] * n_frames
    total, started = 0.0, False
    for f in range(n_frames):
        if f in measured:
            started = True
        total += steps.get(f, 0.0)
        out[f] = round(total, 3) if started else None
    return out


def depth_from_own_baseline(y: float, side: str) -> float:
    """Metres inside the court from the player's own baseline (negative = behind it)."""
    return (COURT_L - y) if side == "near" else y


def heatmap(points: list[tuple[float, float]], cell: float = 1.0,
            x_range=(-4.0, COURT_W + 4.0), y_range=(-4.0, COURT_L + 4.0)) -> dict:
    """Share of measured frames per `cell` x `cell` metre square."""
    nx = int(math.ceil((x_range[1] - x_range[0]) / cell))
    ny = int(math.ceil((y_range[1] - y_range[0]) / cell))
    grid = [[0] * nx for _ in range(ny)]
    for x, y in points:
        i = int((x - x_range[0]) // cell)
        j = int((y - y_range[0]) // cell)
        if 0 <= i < nx and 0 <= j < ny:
            grid[j][i] += 1
    n = max(1, len(points))
    return {"cell_m": cell, "x0": x_range[0], "y0": y_range[0], "nx": nx, "ny": ny,
            "frames": len(points),
            "share": [[round(c / n, 4) for c in row] for row in grid]}


def player_movement(frames: list[dict], pid: str, side: str, fps: float) -> dict:
    """Summary + per-frame series for one player from export_web_data frame records."""
    n = len(frames)
    raw, smooth = [], []
    for r in frames:
        c = (r.get("players", {}).get(pid) or {}).get("court") or {}
        if c.get("status") == "measured" and c.get("xy_m"):
            raw.append((r["f"] - frames[0]["f"], tuple(c["xy_m"])))
            smooth.append((r["f"] - frames[0]["f"], tuple(c.get("xy_m_smoothed") or c["xy_m"])))
    cum_raw = cumulative_distance(raw, n)
    cum_smooth = cumulative_distance(smooth, n)
    depth = [None] * n
    lateral = [None] * n
    for f, (x, y) in smooth:
        depth[f] = round(depth_from_own_baseline(y, side), 3)
        lateral[f] = round(x, 3)
    steps = _steps(smooth)
    gaps = sum(1 for (f0, _), (f1, _) in zip(smooth, smooth[1:]) if f1 - f0 > 1)
    measured_depths = [d for d in depth if d is not None]
    behind = sum(1 for d in measured_depths if d < 0)
    return {
        "side": side,
        "frames_measured": len(raw),
        "frames_total": n,
        "path_m_smoothed": round(sum(s for _, s in steps), 2),
        "path_m_raw": round(sum(s for _, s in _steps(raw)), 2),
        "tracked_seconds": round(len(steps) / fps, 2),
        "gaps_not_bridged": gaps,
        "share_behind_baseline": round(behind / len(measured_depths), 3) if measured_depths else None,
        "series": {"cumulative_m": cum_smooth, "cumulative_m_raw": cum_raw,
                   "depth_m": depth, "lateral_m": lateral},
        "heatmap": heatmap([p for _, p in smooth]),
    }
