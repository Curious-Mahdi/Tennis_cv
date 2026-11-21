"""
scripts/export_web_data.py
──────────────────────────
Per-frame player / ball / court data for a web viewer, produced with the SAME models and
post-processing as main.py, but streamed so it runs on low-memory, CPU-only machines.

Why a separate script instead of main.py
----------------------------------------
main.py holds every decoded frame in memory (read_video) and renders an annotated video.
For a 90 s 1280x720 clip that is ~7.4 GB of frames before any model runs, and it writes
no per-frame positions to disk. This script:

  * decodes one frame at a time and runs the repo's own detectors on it
    (PlayerTracker.detect_frame = YOLOv8x + ByteTrack, CourtLineDetector.predict,
    TrackNet with the same prev/current/next 3-frame window as
    TrackNetBallTracker.detect_frames), so the detections are the ones main.py would get;
  * checkpoints raw detections so a long CPU run can be resumed;
  * then reuses main.py's post-processing functions unchanged (keypoint smoothing, the
    court-validity gate, two-player selection and its sanity check, derive_shot_frames,
    MiniCourt's homography);
  * writes per-frame JSON (+ a flat CSV) with an explicit status on every value.

Ball court position policy (the important one)
----------------------------------------------
The court homography maps the FLOOR plane. An airborne ball projected through it lands
metres away from where it really is, so this script never does that:

  bounce          ball projected at its lowest image point within +/-4 frames of a
                  detected bounce (same rule as main.py's 3-D reconstruction). Measured.
  contact_approx  at a detected racket contact the ball is in the air; the hitter's feet
                  are used as the ground point (same rule as main.py). Approximate.
  interpolated    between two consecutive anchors above, at most MAX_FLIGHT_S apart, the
                  ground track is linearly interpolated. An estimate, not a measurement.
  unavailable     everything else.

Usage
-----
    python scripts/export_web_data.py --input input_videos/tennis_clip.mp4 \
        --end-frame 2335 --out-dir output/web_data

    --end-frame is exclusive. Frames outside [start, end) are still listed in the output,
    marked as outside the analysed segment, so a viewer's timeline covers the whole video.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")   # silence per-frame ultralytics prints

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)   # config model paths are relative to the repo root

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import constants  # noqa: E402
from court_line_detector import CourtLineDetector  # noqa: E402
from main import _lowest_ball_point_near, load_config  # noqa: E402
from mini_visual_court import MiniCourt  # noqa: E402
from trackers import PlayerTracker  # noqa: E402
from trackers.tracknet_ball_tracker import TrackNetBallTracker  # noqa: E402
from utils import (  # noqa: E402
    MIN_LINE_SUPPORT,
    UILayoutManager,
    assess_court_fit_detail,
    assess_selection,
    derive_shot_frames,
    select_two_players,
    smooth_trajectories,
    striking_side,
)
from utils.court_validity import line_support_score  # noqa: E402
from utils.fps_support import assess_fps  # noqa: E402

SCHEMA_VERSION = "1.0"
COURT_LENGTH_M = 2 * constants.HALF_COURT_LINE_HEIGHT        # 23.76
COURT_WIDTH_M = constants.DOUBLE_LINE_WIDTH                    # 10.97
# Players and bounces further than this outside the doubles lines are treated as a
# mapping failure rather than a position (a run-off behind the baseline is ~6.4 m).
PLAYER_MARGIN_SIDE_M, PLAYER_MARGIN_END_M = 6.0, 9.0
BALL_OUT_MARGIN_M = 5.0          # same bound main.py uses for 3-D event endpoints
MAX_BALL_IMAGE_GAP = 10          # longest detection gap shown as "interpolated" (frames)
MAX_FLIGHT_S = 2.0               # longest anchor-to-anchor span interpolated on court
CHECKPOINT_EVERY = 25

log = logging.getLogger("export_web_data")


# ── helpers ────────────────────────────────────────────────────────────────────

def _keep_awake(on: bool) -> None:
    """Stop Windows sleeping mid-run (a multi-hour CPU job). No-op elsewhere."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def _r(v, n=2):
    return None if v is None else round(float(v), n)


def _xy(p, n=2):
    return None if p is None else [_r(p[0], n), _r(p[1], n)]


def _video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count_header": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def _frames(path: str):
    """Sequential decode. Seeking is avoided on purpose: H.264 seeks can land off by a
    few frames, which would silently misalign detections and timestamps."""
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


class _SampledFrames:
    """list-like view exposing only the frames assess_court_fit_detail will sample."""

    def __init__(self, n: int, frames: dict):
        self._n, self._frames = n, frames

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._frames[int(i)]


# ── stage 1: streaming detection ───────────────────────────────────────────────

def detect(args, cfg, seg_start, seg_end, ckpt_path: Path) -> dict:
    n_seg = seg_end - seg_start
    state = {"players": [], "keypoints": [], "ball": [], "resumed_at": [],
             "timing_s": {"yolo": 0.0, "court": 0.0, "tracknet": 0.0}}
    key = {"video": os.path.abspath(args.input), "size": os.path.getsize(args.input),
           "start": seg_start, "end": seg_end}
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            saved = pickle.load(f)
        if saved.get("key") == key:
            state = saved["state"]
            log.info(f"Resuming from checkpoint: {len(state['players'])}/{n_seg} frames done "
                     f"(ball {len(state['ball'])}, one behind by design)")
        else:
            log.warning("Checkpoint belongs to a different video/segment - ignoring it")
    # Players/keypoints are done up to `done`; the ball lags one frame behind because
    # TrackNet needs the NEXT frame before it can run on the current one.
    done = len(state["players"])
    if done >= n_seg and len(state["ball"]) >= n_seg:
        return state

    player_tracker = PlayerTracker(model_path=cfg["models"]["player"])
    court_detector = CourtLineDetector(cfg["models"]["court"])
    ball_tracker = TrackNetBallTracker(model_path=cfg["models"]["tracknet"])
    if ball_tracker.model is None:
        raise SystemExit("TrackNet weights missing - see scripts/download_models.py")

    # A resumed run has a fresh ByteTrack state, so its ids would restart at 1 and
    # collide with earlier tracks. Offset them so no two different tracks share an id;
    # the cost (a player split into two tracks at the resume point) is logged.
    id_offset = 0
    if done:
        seen = [tid for fr in state["players"] for tid in fr]
        id_offset = (max(seen) + 1) if seen else 0
        state["resumed_at"].append(seg_start + done)
        log.warning(f"Track ids offset by {id_offset} after resume at frame "
                    f"{seg_start + done}: tracks crossing this frame are split in two")

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "state": state}, f)
        os.replace(tmp, ckpt_path)

    # TrackNet for frame i needs frames i-1, i, i+1 (clamped at the segment edges,
    # exactly as TrackNetBallTracker.detect_frames clamps at the list edges).
    window: dict = {}   # idx -> frame, holds at most the last 3 decoded frames
    t0 = time.time()
    first_new = seg_start + done

    def run_tracknet(j: int) -> None:
        """Ball for frame j from frames j-1, j, j+1, clamped to the segment."""
        prev_f = window.get(j - 1, window[j]) if j > seg_start else window[j]
        next_f = window.get(j + 1, window[j]) if j < seg_end - 1 else window[j]
        cur_f = window[j]
        t = time.time()
        with torch.no_grad():
            tensor = ball_tracker._preprocess([prev_f, cur_f, next_f])
            out = ball_tracker.model(tensor).argmax(dim=1).detach().cpu().numpy()
        bbox = ball_tracker._output_to_bbox(out[0], cur_f.shape[0], cur_f.shape[1])
        state["timing_s"]["tracknet"] += time.time() - t
        state["ball"].append({1: bbox} if bbox is not None else {})

    ctx_start = max(seg_start, first_new - 2)   # TrackNet context before a resume point
    for idx, frame in _frames(args.input):
        if idx < ctx_start:
            continue
        if idx >= seg_end:
            break
        window[idx] = frame
        window.pop(idx - 3, None)

        if idx >= first_new:
            t = time.time()
            pdict = player_tracker.detect_frame(frame)
            state["timing_s"]["yolo"] += time.time() - t
            state["players"].append({tid + id_offset: box for tid, box in pdict.items()})

            t = time.time()
            state["keypoints"].append(court_detector.predict(frame))
            state["timing_s"]["court"] += time.time() - t

        # The previous frame's ball, now that its next frame is known.
        if idx - 1 >= seg_start and len(state["ball"]) == idx - 1 - seg_start:
            run_tracknet(idx - 1)
        # The segment's last frame has no next frame: clamp, as detect_frames does.
        if idx == seg_end - 1 and len(state["ball"]) == idx - seg_start:
            run_tracknet(idx)

        n = idx - seg_start + 1
        if idx >= first_new and (n % CHECKPOINT_EVERY == 0 or idx == seg_end - 1):
            save()
            rate = (time.time() - t0) / max(1, idx - first_new + 1)
            log.info(f"  frame {n}/{n_seg}  {rate:.2f} s/frame  "
                     f"ETA {rate * (n_seg - n) / 60:.1f} min")

    if len(state["ball"]) != n_seg or len(state["players"]) != n_seg:
        raise SystemExit(f"Decoded fewer frames than expected: ball={len(state['ball'])} "
                         f"players={len(state['players'])} expected={n_seg}")
    save()
    return state


# ── stage 2: post-processing (repo functions) and export ───────────────────────

def build_export(args, cfg, info, seg_start, seg_end, state) -> tuple[dict, list]:
    fps = info["fps"] if info["fps"] and info["fps"] > 0 else 30.0
    n_seg = seg_end - seg_start
    W, H = info["width"], info["height"]

    raw_ball = state["ball"]
    raw_players = state["players"]

    # Same keypoint smoothing as CourtLineDetector.predict_all_frames(smooth=True, 5).
    # smooth_keypoints uses no instance state, so it is called without loading ResNet-50.
    all_kp = CourtLineDetector.smooth_keypoints(None, list(state["keypoints"]), 5)
    kp0 = all_kp[0]

    # One sequential pass: per-frame line support (informational) and the exact frames
    # the clip-level gate samples.
    sample_idx = set(np.linspace(0, n_seg - 1, min(12, n_seg), dtype=int).tolist())
    sampled, support = {}, [None] * n_seg
    first_frame = None
    for idx, frame in _frames(args.input):
        if idx < seg_start:
            continue
        if idx >= seg_end:
            break
        j = idx - seg_start
        if first_frame is None:
            first_frame = frame.copy()
        support[j] = line_support_score(frame, all_kp[j])
        if j in sample_idx:
            sampled[j] = frame.copy()
    court_valid, median_support, court_detail = assess_court_fit_detail(
        _SampledFrames(n_seg, sampled), all_kp)

    # Players: identical selection + sanity gate to main.py.
    people_detected = len({tid for fr in raw_players for tid in fr})
    tracker_for_selection = PlayerTracker.__new__(PlayerTracker)   # scoring only, no model
    players, id_map = select_two_players(tracker_for_selection, raw_players, kp0)
    net_y_img = (kp0[1] + kp0[5]) / 2.0
    selection = assess_selection(players, net_y=net_y_img, people_detected=people_detected)

    # Ball: same interpolation and event derivation as main.py.
    tn = TrackNetBallTracker.__new__(TrackNetBallTracker)            # methods only
    ball_interp = tn.interpolate_ball_positions(raw_ball)
    contacts, bounces, raw_reversals, decode_notes = derive_shot_frames(
        tn, ball_interp, players, cfg.get("detection", {}).get("shot_player_distance_px", 300))

    # Mini-court geometry exactly as main.py lays it out; used for the homography target
    # and the metres-per-pixel scale. Positions are NOT clamped to the drawing panel here
    # (main.py clamps for display), because a player 3 m behind the baseline is real.
    layout = UILayoutManager((H, W, 3), kp0)
    mini = MiniCourt(first_frame, layout_params=layout.get_mini_court_params())
    px_to_m = constants.DOUBLE_LINE_WIDTH / mini.get_width_of_mini_court()
    ox, oy = mini.court_start_x, mini.court_start_y

    h_cache: dict = {}

    def homography(j):
        k = tuple(all_kp[j])
        if k not in h_cache:
            h_cache[k] = mini.compute_homography(all_kp[j])
        return h_cache[k]

    def to_mini(j, pt):
        Hm = homography(j)
        return None if Hm is None else mini.apply_homography(Hm, pt)

    def to_m(mpt):
        return None if mpt is None else ((mpt[0] - ox) * px_to_m, (mpt[1] - oy) * px_to_m)

    def plausible(xy, side, end):
        return (-side <= xy[0] <= COURT_WIDTH_M + side) and (-end <= xy[1] <= COURT_LENGTH_M + end)

    # Player court positions (feet = bottom-centre, the floor-valid point).
    player_mini: dict = {}
    for j in range(n_seg):
        player_mini[j] = {}
        for pid, b in players[j].items():
            m = to_mini(j, ((b[0] + b[2]) / 2.0, float(b[3])))
            if m is not None:
                player_mini[j][pid] = m
    # main.py's Kalman smoother, in mini-court pixels so its tuning applies unchanged.
    player_mini_smooth, _ = smooth_trajectories(player_mini)

    # Ball court anchors.
    bounce_set, contact_set = set(bounces), set(contacts)
    anchors: dict[int, tuple[str, tuple]] = {}
    hitter_by_frame: dict[int, int] = {}
    rejected_events = []
    for f in sorted(bounce_set | contact_set):
        if f in bounce_set:
            pt = _lowest_ball_point_near(ball_interp, f, window=4)
            kind = "bounce"
        else:
            hitter = striking_side(f, ball_interp, players)
            if hitter is None:
                continue
            b = players[f][hitter]
            pt = ((b[0] + b[2]) / 2.0, float(b[3]))
            hitter_by_frame[f] = hitter
            kind = "contact_approx"
        if pt is None:
            continue
        xy = to_m(to_mini(f, pt))
        if xy is None:
            continue
        if plausible(xy, BALL_OUT_MARGIN_M, BALL_OUT_MARGIN_M):
            anchors[f] = (kind, xy)
        else:
            rejected_events.append({"frame": seg_start + f, "kind": kind, "xy_m": _xy(xy)})

    ball_court: dict[int, tuple[str, tuple | None]] = {}
    af = sorted(anchors)
    for f in af:
        ball_court[f] = anchors[f]
    max_span = int(round(MAX_FLIGHT_S * fps))
    for a, b in zip(af, af[1:]):
        if b - a > max_span:
            continue
        (_, pa), (_, pb) = anchors[a], anchors[b]
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            ball_court[f] = ("interpolated", (pa[0] + t * (pb[0] - pa[0]),
                                              pa[1] + t * (pb[1] - pa[1])))

    # Ball image status: raw detection, short-gap interpolation, or unavailable.
    det = [bool(d.get(1)) for d in raw_ball]
    img_status = ["unavailable"] * n_seg
    j = 0
    while j < n_seg:
        if det[j]:
            img_status[j] = "detected"
            j += 1
            continue
        k = j
        while k < n_seg and not det[k]:
            k += 1
        if j > 0 and k < n_seg and (k - j) <= MAX_BALL_IMAGE_GAP:
            for g in range(j, k):
                img_status[g] = "interpolated"
        j = k

    # ── assemble ──
    total = info["frames_decoded"]
    frames_out, csv_rows = [], []
    for fi in range(total):
        t = fi / fps
        rec = {"f": fi, "t": round(t, 4)}
        if not (seg_start <= fi < seg_end):
            rec.update({"in_segment": False, "court": None, "players": {}, "ball": None})
            frames_out.append(rec)
            csv_rows.append({"frame": fi, "t": round(t, 4), "in_segment": 0})
            continue
        j = fi - seg_start
        Hm = homography(j)
        rec["in_segment"] = True
        rec["court"] = {"homography": Hm is not None, "line_support": _r(support[j], 3),
                        "line_support_ok": bool(support[j] is not None
                                                and support[j] >= MIN_LINE_SUPPORT),
                        # Smoothed keypoints (x0, y0, ... x13, y13): what the homography and
                        # analytics.ball_flight's per-frame camera are fitted to.
                        "keypoints": [_r(v, 1) for v in all_kp[j]]}
        row = {"frame": fi, "t": round(t, 4), "in_segment": 1,
               "homography": int(Hm is not None), "line_support": _r(support[j], 3)}

        prec = {}
        for pid in (1, 2):
            b = players[j].get(pid)
            if b is None:
                prec[str(pid)] = {"status": "unavailable"}
                row.update({f"p{pid}_status": "unavailable"})
                continue
            foot = ((b[0] + b[2]) / 2.0, float(b[3]))
            entry = {"status": "detected", "bbox": [_r(v, 1) for v in b],
                     "image_xy": _xy(foot, 1)}
            m = player_mini[j].get(pid)
            if not court_valid:
                court = {"status": "court_calibration_failed", "xy_m": None}
            elif m is None:
                court = {"status": "no_homography", "xy_m": None}
            else:
                xy = to_m(m)
                if plausible(xy, PLAYER_MARGIN_SIDE_M, PLAYER_MARGIN_END_M):
                    court = {"status": "measured", "xy_m": _xy(xy),
                             "xy_m_smoothed": _xy(to_m(player_mini_smooth[j][pid]))}
                else:
                    court = {"status": "outside_plausible_area", "xy_m": None}
            entry["court"] = court
            prec[str(pid)] = entry
            row.update({f"p{pid}_status": "detected", f"p{pid}_img_x": entry["image_xy"][0],
                        f"p{pid}_img_y": entry["image_xy"][1], f"p{pid}_court_status": court["status"],
                        f"p{pid}_x_m": (court["xy_m"] or [None, None])[0],
                        f"p{pid}_y_m": (court["xy_m"] or [None, None])[1]})
        rec["players"] = prec

        st = img_status[j]
        if st == "detected":
            b = raw_ball[j][1]
            ixy = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        elif st == "interpolated":
            b = ball_interp[j][1]
            ixy = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        else:
            ixy = None
        cs, cxy = ball_court.get(j, ("unavailable", None))
        if not court_valid:
            cs, cxy = "court_calibration_failed", None
        event = "bounce" if j in bounce_set else ("contact" if j in contact_set else None)
        rec["ball"] = {"image": {"status": st, "xy": _xy(ixy, 1)},
                       "court": {"status": cs, "xy_m": _xy(cxy)}, "event": event}
        row.update({"ball_img_status": st, "ball_img_x": (rec["ball"]["image"]["xy"] or [None, None])[0],
                    "ball_img_y": (rec["ball"]["image"]["xy"] or [None, None])[1],
                    "ball_court_status": cs, "ball_x_m": (rec["ball"]["court"]["xy_m"] or [None, None])[0],
                    "ball_y_m": (rec["ball"]["court"]["xy_m"] or [None, None])[1], "event": event or ""})
        frames_out.append(rec)
        csv_rows.append(row)

    # Which selected id is the near (bottom of image) player.
    side = {}
    for pid in (1, 2):
        ys = sorted(fr[pid][3] for fr in players if pid in fr)
        if ys:
            side[str(pid)] = "near" if ys[len(ys) // 2] > net_y_img else "far"

    def count(pred):
        return sum(1 for r in frames_out if r.get("in_segment") and pred(r))

    ball_detected = sum(det)
    export = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "video": args.input.replace("\\", "/"),
            "fps": fps, "fps_support": assess_fps(info["fps"]).as_dict(),
            "frame_count": total, "width": W, "height": H,
            "duration_s": round(total / fps, 3),
            "timestamp_rule": "t = f / fps (constant frame rate)",
        },
        "analysis": {
            "segment": {"start_frame": seg_start, "end_frame_exclusive": seg_end,
                        "reason": args.segment_reason},
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "models": {
                "players": f"{cfg['models']['player']} (Ultralytics YOLO) + ByteTrack via model.track(persist=True)",
                "ball": f"TrackNet ({cfg['models']['tracknet']}), 3-frame input, largest-component postprocess",
                "court": f"ResNet-50 14-keypoint regression ({cfg['models']['court']}), per frame, 5-frame moving-average smoothing",
                "court_mapping": "cv2.findHomography RANSAC 10px, 14 keypoints -> MiniCourt reference court",
                "events": "utils.hit_bounce_classifier.derive_shot_frames (3 candidate generators + trajectory classifier + rally decoding)",
            },
            "detector_seconds": {k: round(v, 1) for k, v in state["timing_s"].items()},
            "resumed_at_frames": state.get("resumed_at", []),
        },
        "court": {
            "coordinate_system": {
                "units": "metres", "origin": "far-left corner of the doubles court (as seen from the camera)",
                "x": f"across the court, 0 = far-left doubles sideline, {COURT_WIDTH_M} = right doubles sideline",
                "y": f"along the court, 0 = far baseline, {COURT_LENGTH_M:.2f} = near baseline",
                "net_y": round(COURT_LENGTH_M / 2, 3),
                "singles_x": [constants.DOUBLE_ALLY_DIFFERENCE,
                              round(COURT_WIDTH_M - constants.DOUBLE_ALLY_DIFFERENCE, 2)],
                "service_line_y": [constants.NO_MANS_LAND_HEIGHT,
                                   round(COURT_LENGTH_M - constants.NO_MANS_LAND_HEIGHT, 2)],
            },
            "calibration": {"valid": bool(court_valid), "median_line_support": round(median_support, 3),
                            "threshold": MIN_LINE_SUPPORT, **court_detail},
            "frames_without_homography": sum(1 for j in range(n_seg) if homography(j) is None),
            "frames_line_support_below_threshold": sum(
                1 for s in support if s is not None and s < MIN_LINE_SUPPORT),
        },
        "players": {
            "selection": selection.as_dict(), "selection_reason": selection.reason,
            "track_id_map": {str(k): v for k, v in id_map.items()}, "side": side,
            "frames_detected": {str(p): count(lambda r, p=p: r["players"][str(p)]["status"] == "detected") for p in (1, 2)},
            "frames_court_measured": {str(p): count(lambda r, p=p: r["players"][str(p)].get("court", {}).get("status") == "measured") for p in (1, 2)},
            "court_status_meaning": {
                "measured": "feet (bbox bottom-centre) mapped through that frame's homography; players stand on the floor so this is valid",
                "outside_plausible_area": "mapped more than 6 m beyond a sideline or 9 m beyond a baseline - treated as a mapping failure, withheld",
                "no_homography": "the frame's keypoints could not be fitted, withheld",
                "court_calibration_failed": "the clip failed the court-validity gate, all court positions withheld",
            },
            "xy_m_smoothed": "main.py's Kalman smoother (utils.kalman_smoother) for display; xy_m is the unsmoothed measurement",
        },
        "ball": {
            "raw_detected_frames": ball_detected,
            "raw_coverage": round(ball_detected / n_seg, 3),
            "frames_image_interpolated": img_status.count("interpolated"),
            "frames_image_unavailable": img_status.count("unavailable"),
            "contacts": len(contacts), "bounces": len(bounces),
            "candidate_reversals": len(raw_reversals),
            "court_anchor_frames": {k: sum(1 for v in anchors.values() if v[0] == k)
                                    for k in ("bounce", "contact_approx")},
            "court_frames_interpolated": sum(1 for v in ball_court.values() if v[0] == "interpolated"),
            "events_rejected_outside_court": rejected_events,
            "rally_decoding": getattr(decode_notes, "diagnostics", {}) or {},
            "image_status_meaning": {
                "detected": "TrackNet output on this frame (centre of its 24x24 box)",
                "interpolated": f"gap of <= {MAX_BALL_IMAGE_GAP} frames between detections, linearly interpolated + 3-frame median (main.py's interpolate_ball_positions)",
                "unavailable": "no detection and gap too long (or at the clip edge) to interpolate honestly",
            },
            "court_status_meaning": {
                "bounce": "ball projected at its lowest image point within +/-4 frames of a detected bounce - ball on the floor, homography valid",
                "contact_approx": "ball is in the air at a racket contact; the hitter's feet are used as its ground point (approximate, same rule as main.py 3-D)",
                "interpolated": f"straight-line, constant-speed ground track between two anchors at most {MAX_FLIGHT_S} s apart - an estimate, not a measurement",
                "unavailable": "no reliable ground position; an airborne ball is never projected through the floor homography",
            },
            "known_limits": [
                "README: TrackNet median localisation error 5.4 px, 90th percentile 18 px (TrackNet dataset)",
                "README: contact/bounce recall ~72% on real detections - missed events break the court interpolation into gaps or straight lines through a missed bounce",
                "Ball height is not measured; no 3-D position is exported here",
            ],
        },
        "events": [
            {"frame": seg_start + f, "t": round((seg_start + f) / fps, 4),
             "type": "bounce" if f in bounce_set else "contact",
             **({"hitter": hitter_by_frame[f]} if f in hitter_by_frame else {}),
             "court_xy_m": _xy(anchors[f][1]) if f in anchors else None}
            for f in sorted(bounce_set | contact_set)
        ],
        "frames": frames_out,
    }
    return export, csv_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", "-i", default="input_videos/tennis_clip.mp4")
    ap.add_argument("--out-dir", "-o", default="output/web_data")
    ap.add_argument("--config", "-c", default="configs/config.yaml")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=0, help="exclusive; 0 = end of video")
    ap.add_argument("--segment-reason", default="",
                    help="why the segment was chosen (recorded in the output)")
    ap.add_argument("--tag", default="", help="output file prefix (default: video stem)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.input).stem
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / f"{tag}_export.log", encoding="utf-8")])

    info = _video_info(args.input)
    # Count by decoding, not from the header: the header is an estimate for some files.
    decoded = sum(1 for _ in _frames(args.input))
    info["frames_decoded"] = decoded
    seg_start = max(0, args.start_frame)
    seg_end = min(decoded, args.end_frame or decoded)
    if seg_end - seg_start < 10:
        raise SystemExit("Segment too short")
    log.info(f"Video {args.input}: {decoded} frames decoded ({info['frame_count_header']} in header), "
             f"{info['fps']:.3f} fps, {info['width']}x{info['height']}")
    log.info(f"Analysing frames [{seg_start}, {seg_end}) = {seg_end - seg_start} frames "
             f"on {'cuda' if torch.cuda.is_available() else 'cpu'} "
             f"({torch.get_num_threads()} torch threads)")

    _keep_awake(True)
    t0 = time.time()
    try:
        state = detect(args, cfg, seg_start, seg_end,
                       out / f"{tag}_raw_detections_{seg_start}_{seg_end}.pkl")
        log.info(f"Detection done. Detector seconds: "
                 f"{ {k: round(v) for k, v in state['timing_s'].items()} }")
        log.info("Post-processing with the pipeline's own functions...")
        export, rows = build_export(args, cfg, info, seg_start, seg_end, state)
    finally:
        _keep_awake(False)
    export["analysis"]["wall_clock_s_this_invocation"] = round(time.time() - t0, 1)

    json_path = out / f"{tag}_frames.json"
    json_path.write_text(json.dumps(export, separators=(",", ":"), allow_nan=False),
                         encoding="utf-8")
    summary = {k: v for k, v in export.items() if k != "frames"}
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                             encoding="utf-8")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("frame", "t", "in_segment"), k))
    with open(out / f"{tag}_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    log.info(f"Per-frame JSON -> {json_path}")
    log.info(f"Summary JSON   -> {out / f'{tag}_summary.json'}")
    log.info(f"Per-frame CSV  -> {out / f'{tag}_frames.csv'}")
    c = export["court"]["calibration"]
    log.info(f"Court calibration valid={c['valid']} (median line support {c['median_line_support']})")
    log.info(f"Players: {export['players']['selection']['status']} - {export['players']['selection_reason']}")
    log.info(f"Ball: raw coverage {export['ball']['raw_coverage']:.1%}, "
             f"{export['ball']['contacts']} contacts, {export['ball']['bounces']} bounces")


if __name__ == "__main__":
    main()
