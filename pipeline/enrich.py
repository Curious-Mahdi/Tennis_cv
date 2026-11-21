"""
pipeline/enrich.py
──────────────────
Derived layers added beside an export's per-frame records (moved from
scripts/prepare_web_demo.py; the calculations are unchanged):

  ball.clean    image-space ball track after clean-up      analytics/ball_track.py
  ball.flight   estimated court position of the ball       analytics/ball_flight.py
  analytics     player movement measures                   analytics/movement.py
  ball_clean, ball_flight   summaries of the above

Works on one analysed segment. Frame indices inside the analytics are relative to the
segment start; events and frame records keep their absolute frame numbers.
"""
from __future__ import annotations

import numpy as np

from analytics import clean_ball_track, player_movement
from analytics.ball_flight import cameras, estimate_flights
from analytics.ball_track import MAX_INTERP_GAP, MAX_LINK_GAP, MAX_SPEED_PX, MIN_TRACKLET


def segment_frames(data: dict) -> list[dict]:
    """The analysed (in-segment) frame records, which must be contiguous."""
    seg = [r for r in data["frames"] if r["in_segment"]]
    if not seg:
        raise ValueError("export has no analysed frames")
    if [r["f"] for r in seg] != list(range(seg[0]["f"], seg[-1]["f"] + 1)):
        raise ValueError("analysed frames are not contiguous")
    return seg


def enrich(data: dict, keypoints: list | None = None, log=print) -> tuple[dict, list[dict]]:
    """
    Add the derived layers to `data` (an export_web_data / pipeline.export result).

    keypoints: smoothed court keypoints per analysed frame; taken from the frame records
    when they carry them (exports since keypoints were stored per frame).

    Returns (summary dict without "frames", the analysed frame records, enriched in place).
    """
    seg = segment_frames(data)
    first = seg[0]["f"]
    n = len(seg)
    fps = data["source"]["fps"] if data["source"].get("fps") else 30.0
    if keypoints is None:
        if not all(r.get("court", {}).get("keypoints") for r in seg):
            raise ValueError("export has no per-frame keypoints; pass keypoints=")
        keypoints = [r["court"]["keypoints"] for r in seg]
    if len(keypoints) != n:
        raise ValueError(f"{len(keypoints)} keypoint sets for {n} analysed frames")

    out = {k: v for k, v in data.items() if k != "frames"}
    local_events = [dict(e, frame=e["frame"] - first) for e in data["events"]]

    # Ball: image-space clean-up. Added as a separate `clean` field; the pipeline's own
    # `image` field is left untouched for comparison.
    raw = [tuple(r["ball"]["image"]["xy"]) if r["ball"]["image"]["status"] == "detected" else None
           for r in seg]
    clean = clean_ball_track(raw, [e["frame"] for e in local_events])
    for i, r in enumerate(seg):
        r["ball"]["clean"] = {"status": clean.status[i], "tracklet": clean.tracklet[i],
                              "xy": [round(clean.xy[i][0], 1), round(clean.xy[i][1], 1)] if clean.xy[i] else None}
    out["ball_clean"] = {
        **clean.counts(),
        "rejected_frames": sorted(first + f for t in clean.rejected_tracklets for f in t),
        "params": {"max_speed_px_per_frame": MAX_SPEED_PX, "max_link_gap_frames": MAX_LINK_GAP,
                   "min_tracklet_detections": MIN_TRACKLET, "max_interp_gap_frames": MAX_INTERP_GAP},
        "method": "tracklet association with a speed bound; unconfirmed blips rejected; "
                  "interpolation only inside a tracklet across <= 2 missing frames and never across an event",
    }

    # Ball on the court: estimated flight between detected events.
    cams, focal = cameras(keypoints, data["source"]["width"], data["source"]["height"])
    detected = {i: clean.xy[i] for i in range(n) if clean.status[i] == "detected"}
    hitter_xy = {}
    for e in local_events:
        if e["type"] == "contact" and e.get("hitter"):
            c = (seg[e["frame"]]["players"].get(str(e["hitter"])) or {}).get("court") or {}
            if c.get("xy_m"):
                hitter_xy[e["frame"]] = tuple(c["xy_m"])
    flight = estimate_flights(local_events, detected, cams, fps, n, hitter_xy=hitter_xy)
    for i, r in enumerate(seg):
        r["ball"]["flight"] = flight["frames"].get(i)
    for e in flight["events"]:
        e["frame"] += first
        e["pipeline_frame"] += first
    for s in flight["segments"]:
        s["start"] += first
        s["end"] += first
    kp_err = [c.keypoint_error_px for c in cams]
    out["ball_flight"] = {
        "events": flight["events"], "segments": flight["segments"],
        "camera": {"focal_px": focal, "keypoint_reprojection_median_px": round(float(np.median(kp_err)), 2),
                   "keypoint_reprojection_p95_px": round(float(np.percentile(kp_err, 95)), 2)},
        "frames_with_estimate": len(flight["frames"]),
    }
    log("ball flight: " + str([(s["start"], s["end"], s["status"], s.get("median_residual_px"))
                               for s in flight["segments"]]))

    # Players: movement measures from measured court positions only.
    sides = out.get("players", {}).get("side", {"1": "near", "2": "far"})
    out["analytics"] = {pid: player_movement(seg, pid, sides.get(pid, "near"), fps) for pid in ("1", "2")}
    return out, seg
