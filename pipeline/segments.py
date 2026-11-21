"""
pipeline/segments.py
────────────────────
Find the parts of a video that can be analysed: continuous camera shots that show the
court.

The pipeline assumes one continuous take of a fixed camera (the court-validity gate judges
a clip by its median frame, and tracking breaks at a cut). Broadcast clips and edited
uploads contain cuts to close-ups, replays and graphics. Previously the analysed range was
chosen by hand (--end-frame); here it is found automatically:

1. Shot boundaries: colour-histogram distance between consecutive frames (HSV hue x
   saturation, Bhattacharyya) on downscaled frames. A distance above CUT_THRESHOLD is a
   cut. On the demo clip this finds exactly the two real cuts (frames 2335 and 2387).
2. Court check per shot: COURT_SAMPLES frames per shot are scored with the existing
   court-line support measure (utils.court_validity.line_support_score) on the court
   model's keypoints; a shot is analysable when the median passes MIN_LINE_SUPPORT, the
   same gate build_export applies. Shots shorter than MIN_SEGMENT_S are skipped.

The supported camera is fixed and elevated behind the baseline, so a "cut" here is an
edit, not camera motion. A handheld or panning camera can trigger false cuts; that input
is outside the supported envelope and is reported, not silently split.
"""
from __future__ import annotations

import cv2
import numpy as np

from utils.court_validity import MIN_LINE_SUPPORT, line_support_score

from .errors import AnalysisCancelled

CUT_THRESHOLD = 0.35
MIN_SEGMENT_S = 2.0
COURT_SAMPLES = 5
_THUMB = (160, 90)


def _hist(frame):
    small = cv2.resize(frame, _THUMB, interpolation=cv2.INTER_AREA)
    h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def find_shots(video_path, on_progress=None, cancel=None, total=None) -> tuple[list[tuple[int, int]], list[dict]]:
    """Continuous shots [(start, end_exclusive)] and the detected cuts with their scores."""
    cap = cv2.VideoCapture(str(video_path))
    prev, i, cuts = None, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = _hist(frame)
        if prev is not None:
            d = cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)
            if d > CUT_THRESHOLD:
                cuts.append({"frame": i, "distance": round(float(d), 3)})
        prev = h
        i += 1
        if i % 300 == 0:
            if on_progress and total:
                on_progress(min(1.0, i / total))
            if cancel and cancel():
                cap.release()
                raise AnalysisCancelled("cancelled during shot detection")
    cap.release()
    bounds = [0] + [c["frame"] for c in cuts] + [i]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a], cuts


def court_segments(video_path, shots, models, fps: float) -> list[dict]:
    """Score each shot for a visible court; returns every shot with its verdict."""
    min_len = int(round(MIN_SEGMENT_S * fps))
    wanted = {}
    for k, (a, b) in enumerate(shots):
        if b - a >= min_len:
            for f in np.linspace(a, b - 1, min(COURT_SAMPLES, b - a), dtype=int):
                wanted[int(f)] = k
    scores: dict[int, list[float]] = {}
    if wanted:
        cap = cv2.VideoCapture(str(video_path))
        last = max(wanted)
        i = 0
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                kp = models.court_keypoints(frame)
                scores.setdefault(wanted[i], []).append(line_support_score(frame, kp))
            i += 1
        cap.release()
    out = []
    for k, (a, b) in enumerate(shots):
        s = scores.get(k, [])
        med = float(np.median(s)) if s else None
        if b - a < min_len:
            verdict, reason = False, f"shorter than {MIN_SEGMENT_S:.0f} s"
        elif med is None or med < MIN_LINE_SUPPORT:
            verdict, reason = False, (f"no court in view (line support {med:.2f} < {MIN_LINE_SUPPORT})"
                                      if med is not None else "could not be scored")
        else:
            verdict, reason = True, f"court in view (line support {med:.2f})"
        out.append({"start": a, "end": b, "analysable": verdict, "line_support": None if med is None else round(med, 3),
                    "reason": reason})
    return out
