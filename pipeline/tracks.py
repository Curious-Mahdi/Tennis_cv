"""
pipeline/tracks.py
──────────────────
Re-join player track fragments split by an ID switch.

Why
---
ByteTrack drops a track after a short run of missed detections and starts a new id when
the person is detected again. Player selection (utils.player_selection) keeps ONE track
per player, so every frame under the other fragment is thrown away. On the demo clip the
far player is track 3 for frames 0-10, undetected for frames 11-16, then track 10 from
frame 17, 2 px from where track 3 ended: 11 real frames were discarded. A resume from a
checkpoint splits every track the same way (ids are offset to stay unique).

Rule (deliberately conservative)
--------------------------------
Track B continues track A only if
  * B starts after A ends (no overlap: one person cannot be two tracks at once),
  * the gap is at most MAX_GAP_FRAMES,
  * B's first feet position is within FEET_RADIUS_BOX + FEET_PX_PER_GAP_FRAME * gap of
    A's last (feet = bottom-centre of the box, in pixels),
  * their box heights differ by at most HEIGHT_RATIO.
Joins are one-to-one, cheapest first. Missing frames inside a joined track stay missing;
nothing is interpolated. Every join is reported.
"""
from __future__ import annotations

MAX_GAP_FRAMES = 15             # 0.5 s at 30 fps
FEET_RADIUS_BOX = 0.5           # x box height
FEET_PX_PER_GAP_FRAME = 6.0     # allowance for movement while undetected
HEIGHT_RATIO = 1.5


def _feet(b):
    return (b[0] + b[2]) / 2.0, float(b[3])


def stitch_track_fragments(detections: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Args:
        detections: per-frame {track_id: [x1, y1, x2, y2]} for every detected person.
    Returns:
        (detections with continued tracks renamed to their first id, list of joins).
    """
    spans = {}
    for f, frame in enumerate(detections):
        for tid, box in frame.items():
            s = spans.setdefault(tid, {"first": f, "last": f, "first_box": box, "heights": []})
            s["last"], s["last_box"] = f, box
            s["heights"].append(box[3] - box[1])

    candidates = []
    for a, A in spans.items():
        for b, B in spans.items():
            gap = B["first"] - A["last"] - 1
            if a == b or B["first"] <= A["last"] or gap > MAX_GAP_FRAMES:
                continue
            ha = sum(A["heights"]) / len(A["heights"])
            hb = sum(B["heights"]) / len(B["heights"])
            if max(ha, hb) > HEIGHT_RATIO * min(ha, hb):
                continue
            (xa, ya), (xb, yb) = _feet(A["last_box"]), _feet(B["first_box"])
            dist = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
            if dist <= FEET_RADIUS_BOX * (ha + hb) / 2 + FEET_PX_PER_GAP_FRAME * gap:
                candidates.append((gap, dist, a, b))

    successor, predecessor, joins = {}, {}, []
    for gap, dist, a, b in sorted(candidates):
        if a in successor or b in predecessor:
            continue
        successor[a], predecessor[b] = b, a
        joins.append({"from_track": a, "to_track": b, "gap_frames": gap, "feet_distance_px": round(dist, 1),
                      "at_frame": spans[b]["first"]})

    root = {}
    for tid in spans:
        r = tid
        while r in predecessor:
            r = predecessor[r]
        root[tid] = r
    for j in joins:
        j["merged_into"] = root[j["from_track"]]
    stitched = [{root[tid]: box for tid, box in frame.items()} for frame in detections]
    return stitched, joins
