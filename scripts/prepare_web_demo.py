"""
scripts/prepare_web_demo.py
───────────────────────────
Package an existing export_web_data.py result for the web demo in web/.

  * Cuts the analysed frames out of the source video into a browser-playable H.264 MP4
    whose frame i has presentation time exactly i / fps (same timing as the source), so
    the page can map video.currentTime -> frame number with no offset.
  * Copies the analysed frames' records unchanged (no positions are added or edited).
  * Verifies the alignment by comparing decoded frames of the clip against the source.

    python scripts/prepare_web_demo.py \
        --frames-json output/phase1_test/end_to_end/run150_frames.json \
        --video input_videos/tennis_clip.mp4 --out web
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analytics import clean_ball_track, player_movement  # noqa: E402
from analytics.ball_flight import cameras, estimate_flights  # noqa: E402
from analytics.ball_track import (  # noqa: E402
    MAX_INTERP_GAP, MAX_LINK_GAP, MAX_SPEED_PX, MIN_TRACKLET)


def load_keypoints(seg: list[dict], raw_detections: str | None) -> list:
    """Smoothed court keypoints per analysed frame, from the export or its checkpoint."""
    if all(r.get("court", {}).get("keypoints") for r in seg):
        return [r["court"]["keypoints"] for r in seg]
    if not raw_detections:
        raise SystemExit("This export has no per-frame keypoints; pass --raw-detections <checkpoint.pkl>")
    from court_line_detector import CourtLineDetector
    state = pickle.loads(Path(raw_detections).read_bytes())["state"]
    if len(state["keypoints"]) != len(seg):
        raise SystemExit("--raw-detections does not match this export's frame count")
    # Same smoothing the exporter applies (CourtLineDetector.predict_all_frames).
    return CourtLineDetector.smooth_keypoints(None, list(state["keypoints"]), 5)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="web")
    ap.add_argument("--stills", type=int, nargs="*", default=[106, 52],
                    help="clip frames saved as JPEG stills for the page")
    ap.add_argument("--raw-detections", help="exporter checkpoint .pkl, for exports without per-frame keypoints")
    args = ap.parse_args()

    data = json.loads(Path(args.frames_json).read_text(encoding="utf-8"))
    seg = [r for r in data["frames"] if r["in_segment"]]
    first, last = seg[0]["f"], seg[-1]["f"]
    if first != 0 or [r["f"] for r in seg] != list(range(first, last + 1)):
        raise SystemExit("Analysed segment must be contiguous and start at frame 0")
    n = last + 1

    out = Path(args.out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    clip = out / "media" / "demo_clip.mp4"

    # ── cut + re-encode the first n frames, preserving per-frame timestamps ──
    src = av.open(args.video)
    vs = src.streams.video[0]
    rate = vs.average_rate                               # 30000/1001
    dst = av.open(str(clip), "w", options={"movflags": "+faststart"})
    os_ = dst.add_stream("libx264", rate=rate)
    os_.width, os_.height, os_.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
    # codec time base defaults to 1/rate, so pts = frame index
    os_.options = {"crf": "18", "preset": "medium", "bf": "0", "g": "15"}
    src_frames = []
    for i, frame in enumerate(src.decode(vs)):
        if i >= n:
            break
        img = frame.to_ndarray(format="rgb24")
        if i % max(10, n // 40) == 0 or i == n - 1:      # ~40 alignment samples at most
            src_frames.append((i, img))
        vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        vf.pts = i
        for pkt in os_.encode(vf):
            dst.mux(pkt)
    for pkt in os_.encode():
        dst.mux(pkt)
    dst.close()
    src.close()

    # ── verify: frame count, timestamps, and that clip frame i IS source frame i ──
    chk = av.open(str(clip))
    cs = chk.streams.video[0]
    # Keep only the frames that are compared or saved: a full segment would not fit in RAM.
    keep = {j for i, _ in src_frames for j in (i - 1, i, i + 1)} | set(args.stills)
    times, decoded = [], {}
    for k, fr in enumerate(chk.decode(cs)):
        times.append(fr.time)
        if k in keep:
            decoded[k] = (fr.time, fr.to_ndarray(format="rgb24"))
    chk.close()
    fps = float(rate)
    if len(times) != n:
        raise SystemExit(f"Clip has {len(times)} frames, expected {n}")
    worst_t = max(abs(t - i / fps) for i, t in enumerate(times))
    align = []
    for i, ref in src_frames:
        same = psnr(decoded[i][1], ref)
        nbr = max(psnr(decoded[j][1], ref) for j in (i - 1, i + 1) if 0 <= j < n)
        align.append((i, round(same, 1), round(nbr, 1)))
    misaligned = [a for a in align if a[1] <= a[2]]
    print(f"clip: {clip}  frames={len(times)}  max |pts - i/fps| = {worst_t * 1000:.3f} ms")
    print("alignment (frame, PSNR vs same source frame, best PSNR vs neighbour):", align)
    if misaligned:
        raise SystemExit(f"Alignment check failed: {misaligned}")

    # Real broadcast stills for the page (no stock imagery). Taken from the decoded clip,
    # so they are exactly the frames the analysis describes.
    from PIL import Image
    for f in args.stills:
        if 0 <= f < n:
            Image.fromarray(decoded[f][1]).save(out / "media" / f"still_{f:03d}.jpg", quality=88)

    # ── data: the analysed frames, unchanged, plus derived layers ──
    demo = {k: v for k, v in data.items() if k != "frames"}
    events = [e["frame"] for e in data["events"]]

    # Ball: image-space clean-up (analytics/ball_track.py). Added as a separate `clean`
    # field; the pipeline's own `image` field is left untouched for comparison.
    raw = [tuple(r["ball"]["image"]["xy"]) if r["ball"]["image"]["status"] == "detected" else None
           for r in seg]
    clean = clean_ball_track(raw, events)
    for i, r in enumerate(seg):
        r["ball"]["clean"] = {"status": clean.status[i], "tracklet": clean.tracklet[i],
                              "xy": [round(clean.xy[i][0], 1), round(clean.xy[i][1], 1)] if clean.xy[i] else None}
    demo["ball_clean"] = {
        **clean.counts(),
        "rejected_frames": sorted(f for t in clean.rejected_tracklets for f in t),
        "params": {"max_speed_px_per_frame": MAX_SPEED_PX, "max_link_gap_frames": MAX_LINK_GAP,
                   "min_tracklet_detections": MIN_TRACKLET, "max_interp_gap_frames": MAX_INTERP_GAP},
        "method": "tracklet association with a speed bound; unconfirmed blips rejected; "
                  "interpolation only inside a tracklet across <= 2 missing frames and never across an event",
    }

    # Ball on the court: estimated flight between detected events (analytics/ball_flight.py).
    cams, focal = cameras(load_keypoints(seg, args.raw_detections), data["source"]["width"], data["source"]["height"])
    detected = {i: clean.xy[i] for i in range(n) if clean.status[i] == "detected"}
    hitter_xy = {}
    for e in data["events"]:
        if e["type"] == "contact" and e.get("hitter"):
            c = (seg[e["frame"]]["players"].get(str(e["hitter"])) or {}).get("court") or {}
            if c.get("xy_m"):
                hitter_xy[e["frame"]] = tuple(c["xy_m"])
    flight = estimate_flights(data["events"], detected, cams, fps, n, hitter_xy=hitter_xy)
    for i, r in enumerate(seg):
        r["ball"]["flight"] = flight["frames"].get(i)
    kp_err = [c.keypoint_error_px for c in cams]
    demo["ball_flight"] = {
        "events": flight["events"], "segments": flight["segments"],
        "camera": {"focal_px": focal, "keypoint_reprojection_median_px": round(float(np.median(kp_err)), 2),
                   "keypoint_reprojection_p95_px": round(float(np.percentile(kp_err, 95)), 2)},
        "frames_with_estimate": len(flight["frames"]),
    }
    print("ball flight:", [(s_["start"], s_["end"], s_["status"], s_.get("median_residual_px")) for s_ in flight["segments"]])

    # Players: movement measures from measured court positions only.
    sides = demo.get("players", {}).get("side", {"1": "near", "2": "far"})
    demo["analytics"] = {pid: player_movement(seg, pid, sides.get(pid, "near"), fps) for pid in ("1", "2")}

    demo["frames"] = seg
    demo["demo"] = {
        "video": "media/demo_clip.mp4", "frames": n, "fps": fps,
        "duration_s": n / fps, "source_frames": [first, last],
        "stills": [f"media/still_{f:03d}.jpg" for f in args.stills if 0 <= f < n],
        "timing": "clip frame i has presentation time i/fps, identical to the source video",
        "alignment_check_psnr": [[i, float(a), float(b)] for i, a, b in align],
    }
    (out / "data" / "demo_frames.json").write_text(
        json.dumps(demo, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"data: {out / 'data' / 'demo_frames.json'}  ({n} frames, {len(demo['events'])} events)")


if __name__ == "__main__":
    main()
