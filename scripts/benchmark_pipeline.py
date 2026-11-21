"""
scripts/benchmark_pipeline.py
─────────────────────────────
Measure the real pipeline (pipeline.run.run_analysis) on N frames and estimate the time
for a 10-minute video from the measured per-frame costs.

    python scripts/benchmark_pipeline.py --video input_videos/tennis_clip.mp4 --frames 300

Measured: model load time, time per stage, detection frames/s (with the per-model split),
end-to-end time, peak GPU memory (PyTorch allocator and nvidia-smi), peak system RAM
(sampled every 0.2 s across this process and its children).

Estimated, and labelled so: a 10-minute, 30 fps video (18,000 frames), assuming the whole
video is one analysable segment. Normalisation, shot detection and rendering scale with
the frames of the WHOLE video; detection and post-processing with the ANALYSED frames.
Writes benchmark.json next to the work directory.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil  # noqa: E402

TEN_MIN_FRAMES = 10 * 60 * 30


class PeakRSS:
    """Peak resident memory of this process and its children."""

    def __init__(self, interval=0.2):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        p = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                self.peak = max(self.peak, rss)
            except psutil.Error:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


class PeakNvidiaSmi:
    """Peak GPU memory used on device 0 as nvidia-smi reports it (includes the CUDA context)."""

    def __init__(self, interval=0.5):
        self.interval, self.peak_mb, self._stop = interval, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.available = shutil.which("nvidia-smi") is not None

    def _run(self):
        while self.available and not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split()
                if out:
                    self.peak_mb = max(self.peak_mb or 0, int(out[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input_videos/tennis_clip.mp4")
    ap.add_argument("--frames", type=int, default=300, help="frames to analyse (detection + post-processing)")
    ap.add_argument("--workdir", default="output/benchmark/work")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep an existing work directory (resume instead of re-measure)")
    args = ap.parse_args()

    import torch

    from pipeline.models import ModelBundle
    from pipeline.run import run_analysis

    work = Path(args.workdir)
    if work.exists() and not args.keep:
        shutil.rmtree(work)

    with PeakRSS() as rss, PeakNvidiaSmi() as smi:
        t0 = time.time()
        models = ModelBundle()
        load_s = time.time() - t0
        models.reset_peak()
        t1 = time.time()
        index = run_analysis(args.video, work, models=models, max_frames=args.frames, render=not args.no_render,
                             on_progress=lambda e: None)
        run_s = time.time() - t1
        gpu = models.gpu_memory()

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    stage_s = {k: v.get("seconds", 0.0) for k, v in manifest["stages"].items()}
    video_frames = index["frames"]
    analysed = sum(s["end"] - s["start"] for s in index["segments"])
    det = manifest["stages"]["detect"].get("detector_seconds", {})
    per_frame = {
        "normalize": stage_s.get("normalize", 0) / video_frames,
        "segments": stage_s.get("segments", 0) / video_frames,
        "detect": stage_s.get("detect", 0) / analysed,
        "postprocess": stage_s.get("postprocess", 0) / analysed,
        "render": stage_s.get("render", 0) / video_frames,
    }
    est = {k: v * TEN_MIN_FRAMES for k, v in per_frame.items()}
    result = {
        "measured": {
            "device": models.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "python": platform.python_version(), "cpu": platform.processor(),
            "video_frames": video_frames, "analysed_frames": analysed,
            "model_load_s": round(load_s, 1),
            "stage_seconds": {k: round(v, 1) for k, v in stage_s.items()},
            "detector_seconds": det,
            "detection_fps": round(analysed / stage_s["detect"], 2) if stage_s.get("detect") else None,
            "detector_only_fps": round(analysed / sum(det.values()), 2) if det else None,
            "end_to_end_s": round(run_s, 1),
            "peak_ram_mb": round(rss.peak / 2**20),
            "gpu_memory_torch": gpu,
            "gpu_memory_nvidia_smi_peak_mb": smi.peak_mb,
        },
        "estimate_10_min_video": {
            "assumption": "18,000 frames at 30 fps, all one analysable segment; per-frame costs from this run. "
                          "Linear extrapolation: post-processing has fixed costs (camera fit), so small samples "
                          "overstate it; use >= 300 analysed frames.",
            "seconds_by_stage": {k: round(v) for k, v in est.items()},
            "total_minutes": round(sum(est.values()) / 60, 1),
        },
    }
    out = work.parent / "benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    m, e = result["measured"], result["estimate_10_min_video"]
    print(f"\ndevice {m['device']} {m['gpu'] or ''} | model load {m['model_load_s']} s")
    print(f"analysed {analysed} of {video_frames} frames | stages (s): {m['stage_seconds']}")
    print(f"detection {m['detection_fps']} frames/s (detectors alone {m['detector_only_fps']}); split {det}")
    print(f"peak RAM {m['peak_ram_mb']} MB | GPU torch {gpu} | GPU nvidia-smi peak {m['gpu_memory_nvidia_smi_peak_mb']} MB")
    print(f"ESTIMATE for a 10-minute video: {e['total_minutes']} min  {e['seconds_by_stage']}")
    if analysed < 300:
        print("   (fewer than 300 analysed frames: fixed post-processing costs inflate this estimate)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
