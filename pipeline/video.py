"""
pipeline/video.py
─────────────────
Probe, validate and normalise an uploaded video; cut frame-exact clips.

Normalised form: H.264 MP4, 1280x720, constant frame rate in the 30-fps class, no audio,
frame i at presentation time i / fps. Every later stage reads this file, so analysis
frame numbers, the web player's frame numbers and the annotated video all agree.

Why these choices
-----------------
* Frame rate. Every event threshold in this pipeline is counted in frames and the
  hit/bounce classifier uses pixels-per-frame velocities, measured at 23-31 fps
  (utils/fps_support.py). Videos are brought into that band by keeping every k-th frame
  (k = 1, 2 or 4). Frames are never duplicated: a duplicate is a zero-motion frame that
  the event detector would read as the ball stopping. So 25 fps stays 25 fps (inside the
  measured band), 60 fps becomes 30 fps, 120 fps becomes 30 fps.
* Resolution. Pixel thresholds (ball speed bound, shot proximity) were set at 1280x720,
  so every video is scaled to it. Only 16:9 landscape video is accepted.
* Passthrough. A file already in the normalised form is used byte-for-byte, so results
  on such input are identical to running the scripts on it directly.
"""
from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .errors import AnalysisCancelled, AnalysisError

MAX_DURATION_S = 600.0                 # product limit: 10 minutes
ALLOWED_CONTAINERS = {".mp4", ".mov", ".m4v"}
TARGET_W, TARGET_H = 1280, 720
MIN_W, MIN_H = 960, 540                # below this the far player and ball are too small
ASPECT_RANGE = (1.70, 1.85)            # 16:9 is 1.778
FPS_CLASSES = ((23.0, 31.0, 1), (46.0, 62.0, 2), (92.0, 125.0, 4))   # (min, max, keep every k-th)
X264 = {"crf": "17", "preset": "veryfast", "bf": "0", "g": "30"}


def _rotation(stream) -> int:
    try:
        for sd in getattr(stream, "side_data", {}) or {}:
            if "rotat" in str(sd).lower():
                return int(round(float(stream.side_data[sd]))) % 360
    except Exception:  # noqa: BLE001 - metadata layouts vary between PyAV versions
        pass
    rot = (stream.metadata or {}).get("rotate")
    return int(rot) % 360 if rot else 0


def probe(path) -> dict:
    """Container, stream and timing facts, read from headers plus a short decode."""
    path = Path(path)
    try:
        c = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise AnalysisError("unreadable_video", f"Not a readable video file: {exc}") from exc
    try:
        if not c.streams.video:
            raise AnalysisError("no_video_stream", "The file has no video stream.")
        s = c.streams.video[0]
        rate = s.average_rate or s.guessed_rate
        duration = (float(s.duration * s.time_base) if s.duration
                    else (c.duration / 1e6 if c.duration else None))
        # Frame-interval regularity from the first ~120 frames (variable frame rate check).
        pts = []
        for i, fr in enumerate(c.decode(s)):
            if fr.time is not None:
                pts.append(fr.time)
            if i >= 120:
                break
        d = np.diff(pts) if len(pts) > 2 else np.array([])
        cfr = bool(d.size and float(d.std()) < 0.1 * float(np.median(d)))
        return {
            "container": path.suffix.lower(), "format": c.format.name, "codec": s.codec_context.name,
            "width": s.codec_context.width, "height": s.codec_context.height,
            "pix_fmt": s.codec_context.pix_fmt, "fps": float(rate) if rate else None,
            "rate": str(rate) if rate else None, "duration_s": duration,
            "frames_header": s.frames or None, "rotation": _rotation(s), "constant_frame_rate": cfr,
            "bytes": path.stat().st_size,
        }
    finally:
        c.close()


def plan(info: dict) -> dict:
    """Validate a probe result and decide how to normalise it. Raises AnalysisError."""
    if info["container"] not in ALLOWED_CONTAINERS:
        raise AnalysisError("unsupported_container", f"Upload an MP4 or MOV file (got {info['container'] or 'no extension'}).")
    if info["duration_s"] is None:
        raise AnalysisError("unknown_duration", "The video's duration could not be read.")
    if info["duration_s"] > MAX_DURATION_S + 0.5:
        raise AnalysisError("too_long", f"The video is {info['duration_s'] / 60:.1f} minutes; the limit is "
                                        f"{MAX_DURATION_S / 60:.0f} minutes. Trim it or upload a shorter part.",
                            {"duration_s": round(info["duration_s"], 2), "limit_s": MAX_DURATION_S})
    if info["rotation"]:
        raise AnalysisError("rotated_video", "The video is stored rotated (usually a phone held upright). "
                                             "Record in landscape from a fixed, elevated position behind the baseline.")
    w, h = info["width"], info["height"]
    if not (ASPECT_RANGE[0] <= w / h <= ASPECT_RANGE[1]):
        raise AnalysisError("unsupported_aspect", f"{w}x{h} is not 16:9 landscape video.")
    if w < MIN_W or h < MIN_H:
        raise AnalysisError("resolution_too_low", f"{w}x{h} is below the {MIN_W}x{MIN_H} minimum.")
    fps = info["fps"]
    k = next((k for lo, hi, k in FPS_CLASSES if fps and lo <= fps <= hi), None)
    if k is None:
        raise AnalysisError("unsupported_frame_rate", f"{fps:.1f} fps is not supported. Use 24-30 or 50-60 fps "
                                                      "(or 120 fps slow motion).", {"fps": fps})
    passthrough = (k == 1 and (w, h) == (TARGET_W, TARGET_H) and info["codec"] == "h264"
                   and info["pix_fmt"] == "yuv420p" and info["constant_frame_rate"]
                   and info["container"] in (".mp4", ".m4v"))
    return {"keep_every": k, "out_fps": fps / k, "resize": (w, h) != (TARGET_W, TARGET_H),
            "passthrough": passthrough}


def normalize(src, dst, plan_: dict, on_progress=None, cancel=None) -> dict:
    """
    Write the normalised video to `dst`. Returns metadata including, for every output
    frame, the time of the source frame it came from (`source_time_s`), so analysis
    frames can be mapped back to the original upload.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if plan_["passthrough"]:
        shutil.copyfile(src, dst)
        c = av.open(str(dst))
        times = [fr.time for fr in c.decode(c.streams.video[0])]
        c.close()
        return {"mode": "passthrough", "frames": len(times), "fps": plan_["out_fps"],
                "source_time_s": [round(t, 4) for t in times]}

    k = plan_["keep_every"]
    tmp = dst.with_suffix(".partial.mp4")
    ic = av.open(str(src))
    s = ic.streams.video[0]
    total = s.frames or None
    out_rate = Fraction(s.average_rate or s.guessed_rate) / k
    oc = av.open(str(tmp), "w", options={"movflags": "+faststart"})
    os_ = oc.add_stream("libx264", rate=out_rate)
    os_.width, os_.height, os_.pix_fmt = TARGET_W, TARGET_H, "yuv420p"
    os_.options = dict(X264)
    times, n_out = [], 0
    try:
        for i, fr in enumerate(ic.decode(s)):
            if i % k:
                continue
            out = fr.reformat(width=TARGET_W, height=TARGET_H, format="yuv420p")
            # reformat() keeps the SOURCE frame's time base; pts here counts output frames.
            out.time_base = Fraction(out_rate.denominator, out_rate.numerator)
            out.pts = n_out
            for pkt in os_.encode(out):
                oc.mux(pkt)
            times.append(round(fr.time, 4) if fr.time is not None else None)
            n_out += 1
            if on_progress and total and n_out % 30 == 0:
                on_progress(min(1.0, i / total))
            if cancel and n_out % 30 == 0 and cancel():
                raise AnalysisCancelled("cancelled during normalisation")
        for pkt in os_.encode():
            oc.mux(pkt)
    finally:
        oc.close()
        ic.close()
    tmp.replace(dst)
    return {"mode": "transcoded", "frames": n_out, "fps": float(out_rate), "keep_every": k,
            "resized_to": [TARGET_W, TARGET_H], "source_time_s": times}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def cut_clip(src, dst, start: int, n: int, stills=(), stills_dir=None) -> dict:
    """
    Frames [start, start+n) of `src` as a browser-playable H.264 MP4 whose frame i has
    presentation time exactly i / fps, verified against the source by PSNR (each sampled
    clip frame must match its own source frame better than its neighbours). Optionally
    saves JPEG stills of clip frames. Moved from scripts/prepare_web_demo.py.
    """
    from PIL import Image

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_c = av.open(str(src))
    vs = src_c.streams.video[0]
    rate = vs.average_rate
    dst_c = av.open(str(dst), "w", options={"movflags": "+faststart"})
    os_ = dst_c.add_stream("libx264", rate=rate)
    os_.width, os_.height, os_.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
    os_.options = {"crf": "18", "preset": "medium", "bf": "0", "g": "15"}
    src_frames = []
    for i, frame in enumerate(src_c.decode(vs)):
        if i < start:
            continue
        j = i - start
        if j >= n:
            break
        img = frame.to_ndarray(format="rgb24")
        if j % max(10, n // 40) == 0 or j == n - 1:      # ~40 alignment samples at most
            src_frames.append((j, img))
        vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        vf.pts = j
        for pkt in os_.encode(vf):
            dst_c.mux(pkt)
    for pkt in os_.encode():
        dst_c.mux(pkt)
    dst_c.close()
    src_c.close()

    chk = av.open(str(dst))
    keep = {j for i, _ in src_frames for j in (i - 1, i, i + 1)} | set(stills)
    times, decoded = [], {}
    for k, fr in enumerate(chk.decode(chk.streams.video[0])):
        times.append(fr.time)
        if k in keep:
            decoded[k] = fr.to_ndarray(format="rgb24")
    chk.close()
    fps = float(rate)
    if len(times) != n:
        raise AnalysisError("clip_frame_count", f"Clip has {len(times)} frames, expected {n}")
    worst_t = max(abs(t - i / fps) for i, t in enumerate(times))
    align = []
    for i, ref in src_frames:
        same = psnr(decoded[i], ref)
        nbr = max(psnr(decoded[j], ref) for j in (i - 1, i + 1) if 0 <= j < n)
        align.append((i, round(same, 1), round(nbr, 1)))
    misaligned = [a for a in align if a[1] <= a[2]]
    if misaligned:
        raise AnalysisError("clip_misaligned", f"Alignment check failed: {misaligned}")
    saved = []
    for f in stills:
        if 0 <= f < n and stills_dir is not None:
            p = Path(stills_dir) / f"still_{f:03d}.jpg"
            Image.fromarray(decoded[f]).save(p, quality=88)
            saved.append(p)
    return {"frames": len(times), "fps": fps, "max_pts_error_s": worst_t,
            "alignment_check_psnr": [[i, float(a), float(b)] for i, a, b in align], "stills": saved}
