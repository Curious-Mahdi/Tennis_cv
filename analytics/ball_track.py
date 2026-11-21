"""
analytics/ball_track.py
───────────────────────
Image-space clean-up of TrackNet ball detections, for display and for event derivation.

Why
---
TrackNet emits at most one position per frame and no usable confidence, so a false
positive (a line marking, a shoe, a logo) is indistinguishable from the ball on that frame
alone. The pipeline's own `interpolate_ball_positions` then linearly bridges EVERY gap,
however long, including across false positives. On the 150-frame demo clip that produced a
marker at frame 75 halfway between a false detection at the right edge of the picture and
the real ball near the far baseline - the "teleport" seen in the viewer.

Method (deliberately simple and inspectable)
-------------------------------------------
1. Temporal association. Detections are linked into tracklets: a detection joins an open
   tracklet when it is at most MAX_LINK_GAP frames after that tracklet's last detection
   and within MAX_SPEED_PX * gap pixels of it (a physical speed bound). When several open
   tracklets qualify, the closest per frame wins.
2. False-positive rejection. A tracklet with fewer than MIN_TRACKLET detections is an
   unconfirmed blip and every detection in it is REJECTED (kept in the output, labelled).
3. Short-gap interpolation only. Missing frames are filled ONLY between two consecutive
   detections of the same tracklet, at most MAX_INTERP_GAP frames apart, and never across
   a detected bounce/contact (the trajectory has a corner there; a straight line would cut
   it). Everything else stays UNAVAILABLE.

No smoothing is applied to retained detections: a genuine change of direction at a bounce
or a hit is preserved exactly as detected.

Limits: a false positive that is itself temporally consistent (e.g. a stationary object
detected on several consecutive frames) passes this test. Nothing here can tell those
apart without appearance information.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_SPEED_PX = 70.0     # px/frame at 1280x720; fastest observed real motion ~45 px/frame
MAX_LINK_GAP = 8        # frames; a ball near the apex of a toss can go 6 frames undetected
MIN_TRACKLET = 2        # detections
MAX_INTERP_GAP = 3      # frames between two detections (i.e. <= 2 missing frames)

DETECTED, INTERPOLATED, REJECTED, UNAVAILABLE = "detected", "interpolated", "rejected", "unavailable"


@dataclass
class CleanBallTrack:
    status: list[str]
    xy: list[tuple[float, float] | None]
    tracklet: list[int | None]
    tracklets: list[list[int]] = field(default_factory=list)
    rejected_tracklets: list[list[int]] = field(default_factory=list)

    def counts(self) -> dict:
        raw = sum(1 for s in self.status if s in (DETECTED, REJECTED))
        return {
            "raw_detections": raw,
            "retained": self.status.count(DETECTED),
            "rejected": self.status.count(REJECTED),
            "interpolated": self.status.count(INTERPOLATED),
            "unavailable": self.status.count(UNAVAILABLE),
            "tracklets_retained": len(self.tracklets),
            "tracklets_rejected": len(self.rejected_tracklets),
        }


def associate(raw: list[tuple[float, float] | None],
              max_speed: float = MAX_SPEED_PX, max_gap: int = MAX_LINK_GAP) -> list[list[int]]:
    """Group detection frames into tracklets by a gap + speed bound. Returns frame lists."""
    tracklets: list[list[int]] = []
    for i, p in enumerate(raw):
        if p is None:
            continue
        best, best_cost = None, math.inf
        for t in tracklets:
            j = t[-1]
            gap = i - j
            if gap < 1 or gap > max_gap:
                continue
            q = raw[j]
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            if d <= max_speed * gap and d / gap < best_cost:
                best, best_cost = t, d / gap
        if best is None:
            tracklets.append([i])
        else:
            best.append(i)
    return tracklets


def clean_ball_track(raw: list[tuple[float, float] | None], event_frames=(),
                     max_speed: float = MAX_SPEED_PX, max_gap: int = MAX_LINK_GAP,
                     min_tracklet: int = MIN_TRACKLET,
                     max_interp_gap: int = MAX_INTERP_GAP) -> CleanBallTrack:
    """See module docstring. `raw[i]` is the detected image centre on frame i, or None."""
    n = len(raw)
    events = set(event_frames)
    status = [UNAVAILABLE] * n
    xy: list[tuple[float, float] | None] = [None] * n
    owner: list[int | None] = [None] * n

    kept, dropped = [], []
    for t in associate(raw, max_speed, max_gap):
        (kept if len(t) >= min_tracklet else dropped).append(t)
    for f in (f for t in dropped for f in t):
        status[f] = REJECTED
    for k, t in enumerate(kept):
        for f in t:
            status[f], xy[f], owner[f] = DETECTED, raw[f], k
        for a, b in zip(t, t[1:]):
            if b - a < 2 or b - a > max_interp_gap:
                continue
            if any(a < e < b for e in events):
                continue
            pa, pb = raw[a], raw[b]
            for f in range(a + 1, b):
                if status[f] == REJECTED:
                    continue          # a rejected detection stays visible as rejected
                s = (f - a) / (b - a)
                status[f] = INTERPOLATED
                xy[f] = (pa[0] + s * (pb[0] - pa[0]), pa[1] + s * (pb[1] - pa[1]))
                owner[f] = k
    return CleanBallTrack(status, xy, owner, kept, dropped)
