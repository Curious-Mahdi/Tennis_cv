"""
scripts/prepare_web_demo.py
───────────────────────────
Package an export for the web viewer in web/. Command-line wrapper around the pipeline
package; same flags, same output files, same results (tests/test_pipeline_equivalence.py).

  * pipeline.video.cut_clip   frame-exact H.264 clip of the analysed frames (+ stills),
                              alignment verified against the source by PSNR
  * pipeline.enrich.enrich    ball clean-up, estimated ball flight, movement analytics

    python scripts/prepare_web_demo.py \\
        --frames-json output/web_data/tennis_clip_frames.json \\
        --video input_videos/tennis_clip.mp4 --out web

Exports made before keypoints were stored per frame need --raw-detections (the exporter's
checkpoint .pkl) so the cameras for the flight model can be recovered.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.enrich import enrich, segment_frames  # noqa: E402
from pipeline.video import cut_clip  # noqa: E402


def load_keypoints(seg: list[dict], raw_detections: str | None):
    """Smoothed court keypoints per analysed frame, from the export or its checkpoint."""
    if all(r.get("court", {}).get("keypoints") for r in seg):
        return None                                  # enrich() reads them from the records
    if not raw_detections:
        raise SystemExit("This export has no per-frame keypoints; pass --raw-detections <checkpoint.pkl>")
    from court_line_detector import CourtLineDetector
    state = pickle.loads(Path(raw_detections).read_bytes())["state"]
    if len(state["keypoints"]) != len(seg):
        raise SystemExit("--raw-detections does not match this export's frame count")
    return CourtLineDetector.smooth_keypoints(None, list(state["keypoints"]), 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-json", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="web")
    ap.add_argument("--stills", type=int, nargs="*", default=[106, 52], help="clip frames saved as JPEG stills")
    ap.add_argument("--raw-detections", help="exporter checkpoint .pkl, for exports without per-frame keypoints")
    args = ap.parse_args()

    data = json.loads(Path(args.frames_json).read_text(encoding="utf-8"))
    seg = segment_frames(data)
    first, last = seg[0]["f"], seg[-1]["f"]
    if first != 0:
        raise SystemExit("The web demo expects the analysed segment to start at frame 0")
    n = last + 1

    out = Path(args.out)
    (out / "media").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    clip = cut_clip(args.video, out / "media" / "demo_clip.mp4", 0, n, stills=args.stills, stills_dir=out / "media")
    print(f"clip: {out / 'media' / 'demo_clip.mp4'}  frames={clip['frames']}  "
          f"max |pts - i/fps| = {clip['max_pts_error_s'] * 1000:.3f} ms")
    print("alignment (frame, PSNR vs same source frame, best PSNR vs neighbour):", clip["alignment_check_psnr"])

    demo, seg = enrich(data, keypoints=load_keypoints(seg, args.raw_detections))
    demo["frames"] = seg
    demo["demo"] = {
        "video": "media/demo_clip.mp4", "frames": n, "fps": clip["fps"],
        "duration_s": n / clip["fps"], "source_frames": [first, last],
        "stills": [f"media/still_{f:03d}.jpg" for f in args.stills if 0 <= f < n],
        "timing": "clip frame i has presentation time i/fps, identical to the source video",
        "alignment_check_psnr": clip["alignment_check_psnr"],
    }
    (out / "data" / "demo_frames.json").write_text(json.dumps(demo, separators=(",", ":"), allow_nan=False),
                                                   encoding="utf-8")
    print(f"data: {out / 'data' / 'demo_frames.json'}  ({n} frames, {len(demo['events'])} events)")


if __name__ == "__main__":
    main()
