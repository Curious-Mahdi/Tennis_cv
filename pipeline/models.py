"""
pipeline/models.py
──────────────────
The three detectors, loaded once per worker process and reused for every analysis.

The detectors are the repository's own classes, unchanged:
  players  trackers.PlayerTracker       YOLOv8x + ByteTrack (detect_frame)
  court    CourtLineDetector             ResNet-50, 14 keypoints (predict)
  ball     TrackNetBallTracker           TrackNet, 3-frame input (_preprocess/_output_to_bbox)

Reuse needs one thing the scripts never had to care about: ByteTrack state. Ultralytics
keeps the tracker on the YOLO model's predictor between track() calls, including its
track-id counter, so a second video would continue the first one's tracks. `reset()`
clears it; the pipeline calls it at the start of every segment.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


class ModelsMissing(RuntimeError):
    """A required weight file is absent; see scripts/download_models.py."""


class ModelBundle:
    """Loaded detectors plus per-frame inference with timing. One per worker process."""

    def __init__(self, cfg: dict | None = None):
        import torch

        from court_line_detector import CourtLineDetector
        from trackers import PlayerTracker
        from trackers.tracknet_ball_tracker import TrackNetBallTracker

        from .config import load_config

        self.cfg = cfg or load_config()
        m = self.cfg["models"]
        for key in ("court", "tracknet"):
            if not Path(m[key]).exists():
                raise ModelsMissing(f"{key} weights not found at {m[key]} - run scripts/download_models.py")
        t0 = time.time()
        self.players = PlayerTracker(model_path=m["player"])
        self.court = CourtLineDetector(m["court"])
        self.ball_tracker = TrackNetBallTracker(model_path=m["tracknet"])
        if self.ball_tracker.model is None:
            raise ModelsMissing(f"TrackNet weights could not be loaded from {m['tracknet']}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_seconds = time.time() - t0
        self._torch = torch
        log.info("Models loaded in %.1fs on %s", self.load_seconds, self.device)

    # ── state ──────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget all tracking state: tracks, Kalman filters and the track-id counter."""
        predictor = getattr(self.players.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    # ── per-frame inference (same calls the scripts made) ─────────────────────
    def detect_players(self, frame) -> dict:
        return self.players.detect_frame(frame)

    def court_keypoints(self, frame):
        return self.court.predict(frame)

    def ball(self, prev_frame, frame, next_frame):
        """TrackNet on (prev, cur, next); returns [x1, y1, x2, y2] or None."""
        bt = self.ball_tracker
        with self._torch.no_grad():
            tensor = bt._preprocess([prev_frame, frame, next_frame])
            out = bt.model(tensor).argmax(dim=1).detach().cpu().numpy()
        return bt._output_to_bbox(out[0], frame.shape[0], frame.shape[1])

    def gpu_memory(self) -> dict | None:
        """Peak CUDA memory since the last reset_peak(), in MB (None on CPU)."""
        if self.device != "cuda":
            return None
        c = self._torch.cuda
        return {"peak_allocated_mb": round(c.max_memory_allocated() / 2**20),
                "peak_reserved_mb": round(c.max_memory_reserved() / 2**20)}

    def reset_peak(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()


class LazyModels:
    """Loads a ModelBundle on first use, so work served from checkpoints never loads models."""

    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg
        self._bundle: ModelBundle | None = None

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._bundle is None:
            self._bundle = ModelBundle(self._cfg)
        return getattr(self._bundle, name)
