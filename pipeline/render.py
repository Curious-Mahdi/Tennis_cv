"""
pipeline/render.py
──────────────────
Annotated MP4, streamed: decode one frame, draw, encode, repeat. Memory stays constant
whatever the video length (main.py held every frame in RAM, wrote AVI, and needed ffmpeg
to make it browser-playable; this writes H.264 MP4 directly through PyAV).

Everything drawn comes from the result JSON, with the same rules as the web viewer:
  * players: detector box + label; court position on the inset (feet, measured);
  * ball: the cleaned image track (solid = detected, thin = interpolated), with a trail of
    consecutive frames of the same tracklet only;
  * inset ball: estimated flight hollow, measured bounce solid, nothing when unavailable;
  * frames outside analysed segments are labelled, not annotated.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import cv2

from .errors import AnalysisCancelled

P_COLORS = {"1": (48, 85, 234), "2": (224, 155, 47)}      # BGR of #ea5530, #2f9be0
BALL = (74, 242, 223)                                     # BGR of #dff24a
TRAIL = 8
W_M, L_M, ALLEY, SVC = 10.97, 23.76, 1.37, 5.48
INSET_SCALE = 10.0                                        # px per metre
INSET_PAD_X, INSET_PAD_Y = 2.5, 4.0                       # metres of run-off shown around the court


def _records(segment_files) -> dict[int, dict]:
    recs = {}
    for f in segment_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in data["frames"]:
            recs[r["f"]] = r
        for e in data.get("ball_flight", {}).get("events", []):
            if e["frame"] in recs:
                recs[e["frame"]]["_event"] = e
    return recs


def _inset_size():
    return int((W_M + 2 * INSET_PAD_X) * INSET_SCALE), int((L_M + 2 * INSET_PAD_Y) * INSET_SCALE)


def _inset_origin(width):
    return width - _inset_size()[0] - 16, 16


def _court_px(x0, y0, x, y):
    return int(x0 + (INSET_PAD_X + x) * INSET_SCALE), int(y0 + (INSET_PAD_Y + y) * INSET_SCALE)


def _label(img, text, org, scale=0.55, color=(255, 255, 255)):
    """Text on a dark backing box, legible over any frame."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + base + 4), (40, 24, 12), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_inset(img, rec):
    x0, y0 = _inset_origin(img.shape[1])
    w, h = _inset_size()
    roi = img[y0:y0 + h, x0:x0 + w]
    roi[:] = (roi * 0.25 + (68, 31, 12)).clip(0, 255)        # dark navy panel
    p = lambda x, y: _court_px(x0, y0, x, y)
    white = (245, 245, 245)
    for a, b in [((0, 0), (W_M, 0)), ((0, L_M), (W_M, L_M)), ((0, 0), (0, L_M)), ((W_M, 0), (W_M, L_M)),
                 ((ALLEY, 0), (ALLEY, L_M)), ((W_M - ALLEY, 0), (W_M - ALLEY, L_M)),
                 ((ALLEY, SVC), (W_M - ALLEY, SVC)), ((ALLEY, L_M - SVC), (W_M - ALLEY, L_M - SVC)),
                 ((W_M / 2, SVC), (W_M / 2, L_M - SVC))]:
        cv2.line(img, p(*a), p(*b), white, 1, cv2.LINE_AA)
    cv2.line(img, p(-0.9, L_M / 2), p(W_M + 0.9, L_M / 2), (180, 180, 180), 1, cv2.LINE_AA)
    if rec is None or not rec.get("in_segment"):
        return
    for pid, col in P_COLORS.items():
        c = (rec["players"].get(pid) or {}).get("court") or {}
        if c.get("status") == "measured":
            cv2.circle(img, p(*(c.get("xy_m_smoothed") or c["xy_m"])), 5, col, -1, cv2.LINE_AA)
    fl = rec["ball"].get("flight")
    if fl:
        measured = (rec.get("_event") or {}).get("status") == "measured"
        cv2.circle(img, p(*fl["xy_m"]), 4, BALL, -1 if measured else 1, cv2.LINE_AA)


def _draw(img, f, rec, recs, fps):
    _label(img, f"frame {f}  {f / fps:6.2f} s", (22, 32))
    if rec is None or not rec.get("in_segment"):
        _label(img, "not analysed (camera cut or no court in view)", (22, 64), color=(200, 210, 230))
        _draw_inset(img, None)
        return
    for pid, col in P_COLORS.items():
        pl = rec["players"].get(pid) or {}
        if pl.get("status") == "detected":
            x1, y1, x2, y2 = (int(v) for v in pl["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"P{pid}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    clean = rec["ball"].get("clean") or {}
    for k in range(f - TRAIL + 1, f + 1):
        a, b = recs.get(k - 1), recs.get(k)
        ca = ((a or {}).get("ball") or {}).get("clean") or {}
        cb = ((b or {}).get("ball") or {}).get("clean") or {}
        if ca.get("xy") and cb.get("xy") and ca.get("tracklet") == cb.get("tracklet"):
            cv2.line(img, tuple(int(v) for v in ca["xy"]), tuple(int(v) for v in cb["xy"]), BALL, 2, cv2.LINE_AA)
    if clean.get("xy"):
        cv2.circle(img, tuple(int(v) for v in clean["xy"]), 10, BALL, 2 if clean["status"] == "detected" else 1, cv2.LINE_AA)
    _draw_inset(img, rec)


def render_annotated(video, segment_files, dst, on_progress=None, cancel=None) -> dict:
    recs = _records(segment_files)
    dst = Path(dst)
    tmp = dst.with_suffix(".partial.mp4")
    ic = av.open(str(video))
    s = ic.streams.video[0]
    rate = Fraction(s.average_rate)
    fps = float(rate)
    total = s.frames or None
    oc = av.open(str(tmp), "w", options={"movflags": "+faststart"})
    os_ = oc.add_stream("libx264", rate=rate)
    os_.width, os_.height, os_.pix_fmt = s.codec_context.width, s.codec_context.height, "yuv420p"
    os_.options = {"crf": "20", "preset": "veryfast", "bf": "0", "g": "30"}
    n = 0
    try:
        for f, frame in enumerate(ic.decode(s)):
            img = frame.to_ndarray(format="bgr24")
            _draw(img, f, recs.get(f), recs, fps)
            out = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
            out.pts = f
            for pkt in os_.encode(out):
                oc.mux(pkt)
            n += 1
            if n % 30 == 0:
                if on_progress and total:
                    on_progress(min(1.0, n / total))
                if cancel and cancel():
                    raise AnalysisCancelled("cancelled during rendering")
        for pkt in os_.encode():
            oc.mux(pkt)
    finally:
        oc.close()
        ic.close()
    tmp.replace(dst)
    return {"frames": n, "path": str(dst)}
