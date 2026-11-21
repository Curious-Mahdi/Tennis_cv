"""
scripts/export_web_data.py
──────────────────────────
Per-frame player / ball / court data for one segment of a video, as JSON + CSV.

Command-line wrapper around the pipeline package (pipeline/export.py), which holds the
detection loop and post-processing this script used to contain. Same flags, same output
files, same results (tests/test_pipeline_equivalence.py).

Usage
-----
    python scripts/export_web_data.py --input input_videos/tennis_clip.mp4 \
        --end-frame 2335 --out-dir output/web_data

--end-frame is exclusive. Frames outside [start, end) are still listed in the output,
marked as outside the analysed segment. For whole-video analysis with automatic segment
detection, normalisation and rendering, use pipeline.run.run_analysis instead.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config  # noqa: E402
from pipeline.export import build_export, detect_segment, video_info, write_outputs  # noqa: E402
from pipeline.models import LazyModels  # noqa: E402

log = logging.getLogger("export_web_data")


def _keep_awake(on: bool) -> None:
    """Stop Windows sleeping mid-run (a multi-hour CPU job). No-op elsewhere."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main():
    ap = argparse.ArgumentParser(description="Per-frame data for one segment of a video")
    ap.add_argument("--input", "-i", default="input_videos/tennis_clip.mp4")
    ap.add_argument("--out-dir", "-o", default="output/web_data")
    ap.add_argument("--config", "-c", default=None, help="extra config YAML over configs/config.yaml")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=0, help="exclusive; 0 = end of video")
    ap.add_argument("--segment-reason", default="", help="why the segment was chosen (recorded in the output)")
    ap.add_argument("--tag", default="", help="output file prefix (default: video stem)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.input).stem
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / f"{tag}_export.log", encoding="utf-8")])

    info = video_info(args.input)
    decoded = info["frames_decoded"]
    seg_start = max(0, args.start_frame)
    seg_end = min(decoded, args.end_frame or decoded)
    if seg_end - seg_start < 10:
        raise SystemExit("Segment too short")
    log.info(f"Video {args.input}: {decoded} frames decoded ({info['frame_count_header']} in header), "
             f"{info['fps']:.3f} fps, {info['width']}x{info['height']}")
    log.info(f"Analysing frames [{seg_start}, {seg_end}) = {seg_end - seg_start} frames")

    models = LazyModels(cfg)
    t0, last = time.time(), [0.0]

    def on_frame(done, total):
        if time.time() - last[0] > 30 or done == total:
            last[0] = time.time()
            log.info(f"  frame {done}/{total}")

    _keep_awake(True)
    try:
        state = detect_segment(args.input, seg_start, seg_end,
                               out / f"{tag}_raw_detections_{seg_start}_{seg_end}.pkl", models, on_frame=on_frame)
        log.info(f"Detection done. Detector seconds: { {k: round(v) for k, v in state['timing_s'].items()} }")
        log.info("Post-processing with the pipeline's own functions...")
        export, rows = build_export(args.input, cfg, info, seg_start, seg_end, state,
                                    segment_reason=args.segment_reason)
    finally:
        _keep_awake(False)
    export["analysis"]["wall_clock_s_this_invocation"] = round(time.time() - t0, 1)

    paths = write_outputs(export, rows, out, tag)
    log.info(f"Per-frame JSON -> {paths['frames_json']}")
    log.info(f"Summary JSON   -> {paths['summary_json']}")
    log.info(f"Per-frame CSV  -> {paths['frames_csv']}")
    c = export["court"]["calibration"]
    log.info(f"Court calibration valid={c['valid']} (median line support {c['median_line_support']})")
    log.info(f"Players: {export['players']['selection']['status']} - {export['players']['selection_reason']}")
    log.info(f"Ball: raw coverage {export['ball']['raw_coverage']:.1%}, "
             f"{export['ball']['contacts']} contacts, {export['ball']['bounces']} bounces")


if __name__ == "__main__":
    main()
