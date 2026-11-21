"""
analytics/ball_flight.py
────────────────────────
Estimated ball flight on the court between detected events, checked against the video.

Why the ball cannot simply be projected
---------------------------------------
The court homography maps the ground plane. A ball 2 m in the air projects to a "ground"
point metres beyond where it really is, along the camera ray. Its image position can only
be mapped to the court at the instant it touches the ground.

What this module does
---------------------
1. Camera. A pinhole camera is recovered for every frame from the 14 court keypoints
   (principal point at the image centre, square pixels, one focal length for the clip
   chosen to minimise keypoint reprojection, pose from cv2.solvePnP). Demo clip: keypoints
   reproject to a median 0.43 px, the projected net tape lies on the real one, and a 1.88 m
   vertical at each player's feet matches their box height.

2. Anchors.
   * A bounce is MEASURED only when the video shows the impact: a sharp drop in the ball's
     downward image velocity, with detections on the frames either side (searched within
     ±4 frames of the pipeline's event). Its pixel is mapped to the court (height 0).
   * A bounce the video does not show (ball undetected through it) becomes unknown:
     its frame (searched within ±6 frames) and court position are fitted.
   * A contact keeps its event frame; its strike height is fitted, and its court position
     is where the camera ray through the detected ball reaches that height.

3. Flight model between consecutive anchors:
     ground track  straight line; horizontal speed constant or decaying (one drag-like
                   parameter k per flight, speed x e^-k by the end, k <= 1.2); spin ignored
     height        h(τ) = h_A + (h_B - h_A)·τ + ½·g·T²·τ(1 - τ)      (gravity parabola)
   All unknowns are fitted jointly (scipy least_squares, soft-L1) to the image positions of
   the detected ball, projected through each frame's camera.

4. Check. Each segment is shown only if it has >= 3 detections and its median distance
   between modelled and detected ball is <= MAX_MEDIAN_RESIDUAL_PX. Everything shown is an
   ESTIMATE except measured bounces, and is labelled so.

Limits: height and depth are hard to separate when the ball travels along the camera's
viewing direction; the drag term is a one-parameter approximation; a missed event between two
detected ones breaks the model (the residual check is what catches it); the serve has no
detected contact, so it is not modelled.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

W_M, L_M, ALLEY_M, SVC_M = 10.97, 23.76, 1.37, 5.48
G = 9.81

# 14 court keypoints in metres, in the keypoint model's order (MiniCourt layout).
COURT_KP_M = np.array([
    (0, 0), (W_M, 0), (0, L_M), (W_M, L_M),
    (ALLEY_M, 0), (ALLEY_M, L_M), (W_M - ALLEY_M, 0), (W_M - ALLEY_M, L_M),
    (ALLEY_M, SVC_M), (W_M - ALLEY_M, SVC_M), (ALLEY_M, L_M - SVC_M), (W_M - ALLEY_M, L_M - SVC_M),
    (W_M / 2, SVC_M), (W_M / 2, L_M - SVC_M),
], dtype=np.float64)
_KP3 = np.column_stack([COURT_KP_M, np.zeros(len(COURT_KP_M))])

BOUNCE_SEARCH = 4               # frames either side of the pipeline's bounce
BOUNCE_MIN_VY_DROP = 12.0       # px/frame drop in downward image velocity = visible impact
UNOBSERVED_BOUNCE_SEARCH = 8    # frames either side, when the impact was not seen
MIN_SEGMENT_DETECTIONS = 3
MAX_MEDIAN_RESIDUAL_PX = 10.0
FRAME_GATE_WINDOW = 4           # frames: nearest detection used to gate a single frame
FRAME_GATE_PX = 25.0            # a frame is hidden if that detection disagrees by more
AMBIGUITY_COST_RATIO = 1.25     # fits within 25% of the best mean cost are "equally good"
AMBIGUITY_SPREAD_M = 1.5        # spread above this: shown, but flagged "uncertain" with its radius
MAX_SPREAD_M = 6.0              # spread above this: withheld, the view does not constrain it
MIN_UNCERTAINTY_M = 0.3         # floor (ball localisation error) for any court position
CONTACT_HEIGHT = (0.3, 3.0)     # metres
REACH_M = 1.5                   # a struck ball is within arm + racket of the hitter's feet
REACH_PENALTY_PX_PER_M = 40.0   # weight of the reach constraint, in image-residual units
COURT_MARGIN = 4.0              # metres outside the lines a fitted bounce may land
MAX_DECELERATION = 1.2          # k: ground speed may fall to e^-1.2 = 30% over one flight


# ── camera ────────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    up: float                     # +1/-1: maps "height above court" onto the solver's z axis
    keypoint_error_px: float

    def project(self, pts_xyh) -> np.ndarray:
        """Court (x, y, height) in metres -> image pixels."""
        p = np.asarray(pts_xyh, np.float64).reshape(-1, 3).copy()
        p[:, 2] *= self.up
        out, _ = cv2.projectPoints(p, self.rvec, self.tvec, self.K, None)
        return out.reshape(-1, 2)

    def ray_to_height(self, u: float, v: float, h: float) -> tuple[float, float]:
        """Court (x, y) where the camera ray through pixel (u, v) is h metres above the court."""
        R, _ = cv2.Rodrigues(self.rvec)
        centre = -R.T @ self.tvec.reshape(3)
        d = R.T @ (np.linalg.inv(self.K) @ np.array([u, v, 1.0]))
        p = centre + ((self.up * h - centre[2]) / d[2]) * d
        return float(p[0]), float(p[1])


def _solve(f, k2, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    _, rv, tv = cv2.solvePnP(_KP3, k2, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    pr, _ = cv2.projectPoints(_KP3, rv, tv, K, None)
    return K, rv, tv, float(np.median(np.linalg.norm(pr.reshape(-1, 2) - k2, axis=1)))


def fit_focal(keypoints, width, height, step=10) -> tuple[float, float]:
    """One focal length for the clip: coarse-to-fine scan minimising keypoint reprojection."""
    cx, cy = width / 2.0, height / 2.0
    sample = [np.asarray(k, np.float64).reshape(-1, 2) for k in keypoints[::step]]
    err = lambda f: float(np.median([_solve(f, k, cx, cy)[3] for k in sample]))
    best = min(((f, err(f)) for f in np.arange(600, 6001, 100)), key=lambda t: t[1])
    best = min(((f, err(f)) for f in np.arange(best[0] - 100, best[0] + 101, 10)), key=lambda t: t[1])
    return float(best[0]), best[1]


def cameras(keypoints, width, height) -> tuple[list[Camera], float]:
    """A camera per frame (the broadcast camera pans), sharing one focal length."""
    f, _ = fit_focal(keypoints, width, height)
    cx, cy = width / 2.0, height / 2.0
    cams, up = [], None
    for k in keypoints:
        K, rv, tv, e = _solve(f, np.asarray(k, np.float64).reshape(-1, 2), cx, cy)
        if up is None:   # +height must move a point UP the image
            pr = Camera(K, rv, tv, 1.0, e).project([(W_M / 2, L_M / 2, 0), (W_M / 2, L_M / 2, 1)])
            up = 1.0 if pr[1, 1] < pr[0, 1] else -1.0
        cams.append(Camera(K, rv, tv, up, e))
    return cams, f


# ── anchors ───────────────────────────────────────────────────────────────────

def observed_bounce(frame: int, det: dict) -> tuple[int, float] | None:
    """Frame of the visible impact near `frame` (largest drop in downward image velocity)."""
    best = None
    for f in range(frame - BOUNCE_SEARCH, frame + BOUNCE_SEARCH + 1):
        if all(k in det for k in (f - 1, f, f + 1)):
            drop = (det[f][1] - det[f - 1][1]) - (det[f + 1][1] - det[f][1])
            if drop >= BOUNCE_MIN_VY_DROP and (best is None or drop > best[1]):
                best = (f, drop)
    return best


def _gap_frames(frame: int, det: dict, n_frames: int) -> list[int]:
    """
    Candidate frames for a bounce the video did not show: the run of undetected frames
    nearest the pipeline's event (within UNOBSERVED_BOUNCE_SEARCH). An unseen impact
    happened while the ball was unseen. Letting the fit choose any frame instead biases it
    towards frames that push awkward detections outside every modelled flight.
    """
    lo = max(1, frame - UNOBSERVED_BOUNCE_SEARCH)
    hi = min(n_frames - 2, frame + UNOBSERVED_BOUNCE_SEARCH)
    missing = [f for f in range(lo, hi + 1) if f not in det]
    if not missing:
        return [frame]
    runs, cur = [], [missing[0]]
    for f in missing[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            runs.append(cur); cur = [f]
    runs.append(cur)
    return min(runs, key=lambda r: min(abs(f - frame) for f in r))


def _contact_pixel(frame, det):
    for off in (0, -1, 1, -2, 2):
        if frame + off in det:
            return det[frame + off]
    return None


def _progress(tau, k):
    """Share of the ground track covered at time fraction tau. k = 0 is constant speed;
    k > 0 decelerates (air drag): speed falls by a factor e^-k over the flight."""
    return tau if k < 1e-6 else (1 - np.exp(-k * tau)) / (1 - np.exp(-k))


def _flight(f, a, b, fps, k=0.0):
    """(x, y, h) at frame f between anchors a, b = (frame, x, y, h); k = deceleration."""
    fa, xa, ya, ha = a
    fb, xb, yb, hb = b
    tau = (f - fa) / (fb - fa)
    s = _progress(tau, k)
    T = (fb - fa) / fps
    return (xa + s * (xb - xa), ya + s * (yb - ya),
            ha + tau * (hb - ha) + 0.5 * G * T * T * tau * (1 - tau))


# ── fit ───────────────────────────────────────────────────────────────────────

def estimate_flights(events, det, cams, fps, n_frames, hitter_xy=None, max_flight_s=2.0) -> dict:
    """
    Args:
        events:    pipeline events [{"frame", "type": "bounce"|"contact", "hitter"?}].
        det:       {frame: (u, v)} DETECTED ball pixels (cleaned track; no interpolated frames).
        cams:      per-frame cameras.
        n_frames:  clip length.
        hitter_xy: {contact frame: (x, y)} measured court position of the hitter's feet. The
                   ball at contact must be within REACH_M of them; without this, height and
                   depth trade off along the camera ray and the fit can place a contact
                   metres in front of the player.

    Returns:
        {"events": [...refined anchors...], "segments": [...], "frames": {f: {...}}}
    """
    ev = sorted(events, key=lambda e: e["frame"])
    anchors = []
    for e in ev:
        a = {"type": e["type"], "pipeline_frame": e["frame"]}
        if e["type"] == "bounce":
            ob = observed_bounce(e["frame"], det)
            if ob:
                f = ob[0]
                x, y = cams[f].ray_to_height(*det[f], 0.0)
                a.update(frame=f, status="measured", xy=(x, y), h=0.0, vy_drop_px=round(ob[1], 1))
            else:
                a.update(frame=e["frame"], status="unobserved")
        else:
            px = _contact_pixel(e["frame"], det)
            a.update(frame=e["frame"], status="contact" if px else "no_ball", pixel=px,
                     hitter=e.get("hitter"))
        anchors.append(a)

    pairs = [(i, i + 1) for i in range(len(anchors) - 1)
             if 0 < anchors[i + 1]["frame"] - anchors[i]["frame"] <= max_flight_s * fps
             and anchors[i]["status"] != "no_ball" and anchors[i + 1]["status"] != "no_ball"]
    involved = sorted({i for p in pairs for i in p})
    free_c = [i for i in involved if anchors[i]["status"] == "contact"]
    free_b = [i for i in involved if anchors[i]["status"] == "unobserved"]

    def unpack(params, bframes):
        out = {}
        k = 0
        for i in free_c:
            px = anchors[i]["pixel"]
            h = params[k]; k += 1
            x, y = cams[anchors[i]["frame"]].ray_to_height(px[0], px[1], h)
            out[i] = (anchors[i]["frame"], x, y, h)
        for i, fb in zip(free_b, bframes):
            out[i] = (fb, params[k], params[k + 1], 0.0); k += 2
        for i in involved:
            if anchors[i]["status"] == "measured":
                out[i] = (anchors[i]["frame"], *anchors[i]["xy"], 0.0)
        out["k"] = list(params[k:k + len(pairs)])
        return out

    def residuals(params, bframes, per_seg=False):
        A = unpack(params, bframes)
        res, seg = [], {}
        for n, (i, j) in enumerate(pairs):
            a, b = A[i], A[j]
            if not a[0] < b[0]:
                res.append(1e3)
                continue
            seg[(i, j)] = []
            for f in range(int(a[0]) + 1, int(b[0])):
                if f in det:
                    p = cams[f].project([_flight(f, a, b, fps, A["k"][n])])[0]
                    r = p - np.asarray(det[f])
                    res.extend(r)
                    seg[(i, j)].append((f, float(np.hypot(*r))))
        for i in free_c:                       # physical constraint: within reach
            feet = (hitter_xy or {}).get(anchors[i]["frame"])
            if feet is not None:
                d = float(np.hypot(A[i][1] - feet[0], A[i][2] - feet[1]))
                res.append(max(0.0, d - REACH_M) * REACH_PENALTY_PX_PER_M)
        return seg if per_seg else np.asarray(res if res else [0.0])

    x0, lo, hi = [], [], []
    for _ in free_c:
        x0.append(1.0); lo.append(CONTACT_HEIGHT[0]); hi.append(CONTACT_HEIGHT[1])
    for _ in free_b:
        x0 += [W_M / 2, L_M / 2]; lo += [-COURT_MARGIN, -COURT_MARGIN]; hi += [W_M + COURT_MARGIN, L_M + COURT_MARGIN]
    for _ in pairs:
        x0.append(0.2); lo.append(0.0); hi.append(MAX_DECELERATION)

    grids = [_gap_frames(anchors[i]["frame"], det, n_frames) for i in free_b]
    fits = []
    for bframes in itertools.product(*grids) if grids else [()]:
        if not x0:
            break
        fit = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=5.0, args=(bframes,))
        fits.append((fit.cost / max(1, len(fit.fun)), fit.x, bframes))   # mean cost: comparable
    best = min(fits, key=lambda t: t[0]) if fits else None

    params, bframes = (best[1], best[2]) if best else (np.array([]), ())
    A = unpack(params, bframes) if involved else {}
    per_seg = residuals(params, bframes, per_seg=True) if pairs else {}

    # Uncertainty: every candidate fit almost as good as the best is an equally valid
    # explanation of the video. The spread of the positions they give an anchor is its
    # uncertainty. It is large for an unseen bounce on a ball travelling along the camera's
    # line of sight, where height and depth trade off. Measured bounces get a floor value
    # for localisation error.
    near_best = [f for f in fits if f[0] <= best[0] * AMBIGUITY_COST_RATIO] if best else []
    radius: dict[int, float] = {}
    for i in involved:
        if anchors[i]["status"] == "measured":
            radius[i] = MIN_UNCERTAINTY_M
            continue
        pts = [unpack(x, bf)[i][1:3] for _, x, bf in near_best]
        spread = max((float(np.hypot(p[0] - q[0], p[1] - q[1])) for p in pts for q in pts), default=0.0)
        radius[i] = max(MIN_UNCERTAINTY_M, spread / 2)
    ambiguous = {i: 2 * r for i, r in radius.items() if 2 * r > MAX_SPREAD_M}

    segments, frames = [], {}
    shown_anchor = set()
    for k, (i, j) in enumerate(pairs):
        rs = per_seg.get((i, j), [])
        med = float(np.median([r for _, r in rs])) if rs else None
        seg = {"index": k, "start": int(A[i][0]), "end": int(A[j][0]),
               "from": anchors[i]["type"], "to": anchors[j]["type"],
               "deceleration_k": round(float(A["k"][k]), 2) if pairs else None,
               "detections": len(rs), "median_residual_px": None if med is None else round(med, 1),
               "max_residual_px": round(max(r for _, r in rs), 1) if rs else None}
        amb = [ambiguous[x] for x in (i, j) if x in ambiguous]
        if amb:
            seg.update(status="withheld", reason=(
                f"position not determined by this camera view: equally good fits place it "
                f"up to {max(amb):.1f} m apart"))
        elif len(rs) < MIN_SEGMENT_DETECTIONS:
            seg.update(status="withheld", reason=f"only {len(rs)} ball detections to check the flight against")
        elif med > MAX_MEDIAN_RESIDUAL_PX:
            seg.update(status="withheld", reason=f"modelled flight disagrees with the video (median {med:.0f} px)")
        else:
            ra, rb = radius[i], radius[j]
            seg["status"] = "shown" if max(ra, rb) * 2 <= AMBIGUITY_SPREAD_M else "shown_uncertain"
            seg["uncertainty_m"] = [round(ra, 2), round(rb, 2)]
            shown_anchor |= {i, j}
            rmap = dict(rs)
            a, b = A[i], A[j]
            hidden = 0
            for f in range(int(a[0]), int(b[0]) + 1):
                # Per-frame gate: a frame whose nearest detection contradicts the model is
                # not shown, even when the flight as a whole fits.
                near = [g for g in rmap if abs(g - f) <= FRAME_GATE_WINDOW]
                if near and rmap[min(near, key=lambda g: abs(g - f))] > FRAME_GATE_PX:
                    hidden += 1
                    continue
                x, y, h = _flight(f, a, b, fps, A["k"][k])
                tau = (f - a[0]) / (b[0] - a[0])
                frames[f] = {"xy_m": [round(x, 2), round(y, 2)], "height_m": round(max(h, 0.0), 2),
                             "uncertainty_m": round(ra + tau * (rb - ra), 2), "segment": k,
                             **({"residual_px": round(rmap[f], 1)} if f in rmap else {})}
            seg["frames_hidden_by_gate"] = hidden
        segments.append(seg)

    out_events = []
    for i, a in enumerate(anchors):
        e = {"type": a["type"], "frame": int(A[i][0]) if i in A else a["frame"],
             "pipeline_frame": a["pipeline_frame"], **({"hitter": a["hitter"]} if a.get("hitter") else {})}
        if a["status"] == "measured":
            e.update(status="measured", xy_m=[round(a["xy"][0], 2), round(a["xy"][1], 2)], vy_drop_px=a["vy_drop_px"])
        elif i in shown_anchor:
            e.update(status="estimated", xy_m=[round(float(A[i][1]), 2), round(float(A[i][2]), 2)],
                     uncertainty_m=round(radius[i], 2))
            if a["type"] == "contact":
                e["height_m"] = round(float(A[i][3]), 2)
                feet = (hitter_xy or {}).get(a["frame"])
                if feet is not None:
                    e["distance_to_hitter_m"] = round(float(np.hypot(A[i][1] - feet[0], A[i][2] - feet[1])), 2)
        else:
            e.update(status="unavailable")
        out_events.append(e)
    return {"events": out_events, "segments": segments, "frames": frames}
