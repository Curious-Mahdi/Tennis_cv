"""
Test fixtures for the pipeline: a synthetic court video with known geometry, and a fake
model bundle with the ModelBundle interface. Lets the orchestration (normalisation,
segments, detection loop, checkpoints, post-processing, rendering) run with real code but
without model weights or broadcast footage.
"""
from __future__ import annotations

from fractions import Fraction

import av
import cv2
import numpy as np

W_M, L_M, ALLEY, SVC = 10.97, 23.76, 1.37, 5.48
KP_M = np.array([(0, 0), (W_M, 0), (0, L_M), (W_M, L_M), (ALLEY, 0), (ALLEY, L_M), (W_M - ALLEY, 0),
                 (W_M - ALLEY, L_M), (ALLEY, SVC), (W_M - ALLEY, SVC), (ALLEY, L_M - SVC),
                 (W_M - ALLEY, L_M - SVC), (W_M / 2, SVC), (W_M / 2, L_M - SVC)], np.float32)
LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def court_homography(width=1280, height=720):
    """Court metres -> image pixels for a broadcast-like view at the given size."""
    sx, sy = width / 1280, height / 720
    src = np.float32([(0, 0), (W_M, 0), (0, L_M), (W_M, L_M)])
    dst = np.float32([(450 * sx, 160 * sy), (830 * sx, 160 * sy), (200 * sx, 600 * sy), (1080 * sx, 600 * sy)])
    return cv2.getPerspectiveTransform(src, dst)


def keypoints(width=1280, height=720) -> np.ndarray:
    H = court_homography(width, height)
    return cv2.perspectiveTransform(KP_M.reshape(-1, 1, 2), H).reshape(-1)


def court_frame(width=1280, height=720, tint=0) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (90 + tint, 120, 60)                                       # BGR surround
    kp = keypoints(width, height).reshape(-1, 2)
    cv2.fillPoly(img, [kp[[0, 1, 3, 2]].astype(np.int32)], (150, 80, 40))
    for a, b in LINES:
        cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), (255, 255, 255), 3)
    return img


def write_video(path, n_frames, fps=30, width=1280, height=720, cut_at=None, frame_fn=None):
    """H.264 MP4. Frames after `cut_at` are a flat colour (a camera cut to a graphic)."""
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
    s.width, s.height, s.pix_fmt = width, height, "yuv420p"
    s.options = {"crf": "18", "bf": "0"}
    base = court_frame(width, height)
    for i in range(n_frames):
        if cut_at is not None and i >= cut_at:
            img = np.full((height, width, 3), (30, 200, 240), np.uint8)
        else:
            img = frame_fn(i, base.copy()) if frame_fn else base.copy()
            cv2.putText(img, str(i), (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        vf = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
        vf.pts = i
        for pkt in s.encode(vf):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


class FakeModels:
    """ModelBundle interface with deterministic outputs derived from the frame number.

    The frame number is read from the frame itself (the digits drawn by write_video are
    not parsed; instead each call advances a counter that reset() rewinds), so outputs
    match across runs and across a resume.
    """

    def __init__(self, drop_far=range(20, 26)):
        self.drop_far = set(drop_far)
        self.kp = keypoints()
        self.loads = 1
        self.resets = 0
        self.court_calls = 0

    def reset(self):
        self.resets += 1

    @staticmethod
    def _frame_no(frame) -> int:
        digit = lambda x0: int(min(15, max(0, round((float(frame[4:28, x0 + 4:x0 + 28].mean()) - 8) / 16))))
        return 16 * digit(0) + digit(32)             # encoded by number_frames()

    def detect_players(self, frame):
        f = self._frame_no(frame)
        near_x = 500 + 3 * f
        out = {1: [near_x - 30, 470.0, near_x + 30, 590.0]}
        if f not in self.drop_far:
            far_x = 620 + 2 * f
            # like ByteTrack: after the dropout the far player comes back under a new id
            out[7 if f < min(self.drop_far, default=10**9) else 8] = [far_x - 15, 100.0, far_x + 15, 158.0]
        out[9] = [40.0, 300.0, 70.0, 360.0]                          # a line judge
        return out

    def court_keypoints(self, frame):
        self.court_calls += 1
        return self.kp.copy()

    def ball(self, prev_frame, frame, next_frame):
        f = self._frame_no(frame)
        if f % 7 == 3:
            return None
        x, y = 640 + 4 * f, 200 + 3 * f
        return [x - 12, y - 12, x + 12, y + 12]


def number_frames(i, img):
    """frame_fn for write_video: the frame number as two base-16 digits in grey blocks
    (levels 8, 24, ..., 248), which survive H.264's lossy, limited-range encoding."""
    assert i < 256
    for x0, d in ((0, i // 16), (32, i % 16)):
        v = 8 + 16 * d
        img[0:32, x0:x0 + 32] = (v, v, v)
    return img
