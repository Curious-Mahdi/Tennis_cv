"""
pipeline/run.py
───────────────
run_analysis(): one uploaded video in, one isolated work directory out.

    probe -> normalize -> segments -> detect (per segment) -> postprocess -> render

Work directory (everything an analysis writes lives here; nothing else is touched):

    manifest.json              stage status, timings, versions, error
    normalized.mp4             the analysed video (frame i at i / fps); also the web clip
    normalize.json             probe facts, normalisation plan, source time of each frame
    segments.json              shots found, and which are analysable
    checkpoints/seg_A_B.pkl    raw detections per segment (resumable)
    results/analysis.json      index: source, normalisation, segments, file names
    results/segment_A_B.json   per-segment result in the web viewer's format
    results/segment_A_B.csv    flat per-frame table
    results/annotated.mp4      overlay video (optional)

Recovery: a stage marked complete in the manifest with its outputs present is skipped, and
detection resumes from its checkpoint, so re-running the same call after a crash or a
cancellation continues where it stopped.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .errors import AnalysisCancelled, AnalysisError

log = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"
STAGES = ("probe", "normalize", "segments", "detect", "postprocess", "render")
# Share of the overall progress bar per stage (detection dominates real runtime).
WEIGHTS = {"probe": 0.01, "normalize": 0.07, "segments": 0.05, "detect": 0.77, "postprocess": 0.05, "render": 0.05}


class _Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                     else {"pipeline_version": PIPELINE_VERSION, "stages": {}, "status": "created"})

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def done(self, stage, *paths) -> bool:
        return self.data["stages"].get(stage, {}).get("status") == "done" and all(Path(p).exists() for p in paths)

    def mark(self, stage, status, **extra):
        s = self.data["stages"].setdefault(stage, {})
        s.update(status=status, **extra)
        if status == "running":
            s["started_at"] = time.time()
        elif "started_at" in s:
            s["seconds"] = round(time.time() - s["started_at"], 2)
        self.save()


def _cancel_fn(cancel):
    if cancel is None:
        return lambda: False
    if hasattr(cancel, "is_set"):          # threading.Event / multiprocessing.Event
        return cancel.is_set
    return cancel


def _write_json(path: Path, obj, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, allow_nan=False, **({"separators": (",", ":")} if compact else {"indent": 2}))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_analysis(video, workdir, on_progress=None, cancel=None, models=None, cfg=None,
                 render: bool = True, stitch_tracks: bool = True, max_frames: int | None = None) -> dict:
    """
    Analyse `video` into `workdir`. Returns the results index (results/analysis.json).

    on_progress(dict): {"stage", "stage_progress", "progress", "message"}; progress in [0, 1].
    cancel: callable returning True, or an Event; checked between frames. On cancel,
            AnalysisCancelled is raised and the work directory keeps everything finished.
    models: a loaded pipeline.models.ModelBundle (one per worker). Loaded lazily if None.
    max_frames: analyse at most this many frames in total (benchmarks, previews).

    Raises AnalysisError (with a stable .code) for input the pipeline cannot analyse.
    """
    from . import segments as seg_mod
    from . import video as vid
    from .config import load_config
    from .enrich import enrich
    from .export import build_export, detect_segment, video_info, write_csv
    from .models import LazyModels
    from .tracks import stitch_track_fragments

    video, workdir = Path(video), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (workdir / "results").mkdir(parents=True, exist_ok=True)
    cfg = cfg or load_config()
    models = models if models is not None else LazyModels(cfg)
    is_cancelled = _cancel_fn(cancel)
    m = _Manifest(workdir / "manifest.json")
    m.data.update(status="running", input={"name": video.name, "bytes": video.stat().st_size})
    m.save()

    base = {s: sum(WEIGHTS[x] for x in STAGES[:i]) for i, s in enumerate(STAGES)}

    def report(stage, frac, message=""):
        if on_progress:
            on_progress({"stage": stage, "stage_progress": round(frac, 4),
                         "progress": round(base[stage] + WEIGHTS[stage] * frac, 4), "message": message})

    def check_cancel():
        if is_cancelled():
            raise AnalysisCancelled("cancelled")

    norm_path = workdir / "normalized.mp4"
    try:
        # ── probe + validate ──────────────────────────────────────────────────
        if not m.done("probe", workdir / "normalize.json"):
            m.mark("probe", "running")
            report("probe", 0.0, "reading video")
            info = vid.probe(video)
            plan = vid.plan(info)                       # raises AnalysisError on bad input
            _write_json(workdir / "normalize.json", {"probe": info, "plan": plan})
            m.mark("probe", "done")
        norm_meta = json.loads((workdir / "normalize.json").read_text(encoding="utf-8"))
        report("probe", 1.0)
        check_cancel()

        # ── normalize ─────────────────────────────────────────────────────────
        if not m.done("normalize", norm_path):
            m.mark("normalize", "running")
            res = vid.normalize(video, norm_path, norm_meta["plan"],
                                on_progress=lambda f: report("normalize", f, "normalising video"), cancel=is_cancelled)
            norm_meta["result"] = res
            _write_json(workdir / "normalize.json", norm_meta)
            m.mark("normalize", "done", mode=res["mode"], frames=res["frames"])
        report("normalize", 1.0)
        check_cancel()

        info = video_info(norm_path)
        fps = info["fps"] or 30.0
        # ── segments ──────────────────────────────────────────────────────────
        if not m.done("segments", workdir / "segments.json"):
            m.mark("segments", "running")
            shots, cuts = seg_mod.find_shots(norm_path, total=info["frames_decoded"], cancel=is_cancelled,
                                             on_progress=lambda f: report("segments", 0.8 * f, "finding camera cuts"))
            scored = seg_mod.court_segments(norm_path, shots, models, fps)
            _write_json(workdir / "segments.json", {"cuts": cuts, "shots": scored})
            m.mark("segments", "done")
        shots = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))["shots"]
        todo = [(s["start"], s["end"]) for s in shots if s["analysable"]]
        if not todo:
            raise AnalysisError("no_court_segment", "No part of the video shows the court from a supported "
                                "camera position (fixed, elevated, behind the baseline).",
                                {"shots": [{k: s[k] for k in ("start", "end", "reason")} for s in shots]})
        if max_frames:
            budget, limited = max_frames, []
            for a, b in todo:
                if budget <= 0:
                    break
                limited.append((a, min(b, a + budget)))
                budget -= limited[-1][1] - a
            todo = limited
        report("segments", 1.0)
        check_cancel()

        # ── detect ────────────────────────────────────────────────────────────
        total = sum(b - a for a, b in todo)
        before = {seg: 0 for seg in todo}
        states = {}
        m.mark("detect", "running", frames=total)
        for a, b in todo:
            def on_frame(done, n, _a=a, _b=b):
                before[(_a, _b)] = done
                report("detect", sum(before.values()) / total, f"analysing frames {_a + done}/{_b}")
            states[(a, b)] = detect_segment(norm_path, a, b, workdir / "checkpoints" / f"seg_{a}_{b}.pkl",
                                            models, on_frame=on_frame, cancel=is_cancelled)
            before[(a, b)] = b - a
        m.mark("detect", "done",
               detector_seconds={k: round(sum(s["timing_s"][k] for s in states.values()), 1)
                                 for k in ("yolo", "court", "tracknet")})
        report("detect", 1.0)
        check_cancel()

        # ── postprocess ───────────────────────────────────────────────────────
        results = []
        m.mark("postprocess", "running")
        for k, (a, b) in enumerate(todo):
            state = states[(a, b)]
            joins = []
            if stitch_tracks:
                players, joins = stitch_track_fragments(state["players"])
                state = dict(state, players=players)
            export, rows = build_export(norm_path, cfg, info, a, b, state,
                                        segment_reason="automatic: continuous shot with the court in view")
            export["players"]["track_stitching"] = {"enabled": stitch_tracks, "joins": joins}
            enriched, seg_frames = enrich(export, log=log.info)
            enriched["frames"] = seg_frames
            name = f"segment_{a}_{b}"
            _write_json(workdir / "results" / f"{name}.json", enriched, compact=True)
            write_csv(rows, workdir / "results" / f"{name}.csv")
            results.append({"start": a, "end": b, "file": f"{name}.json", "csv": f"{name}.csv",
                            "court_calibrated": export["court"]["calibration"]["valid"],
                            "players": export["players"]["selection"]["status"],
                            "events": len(enriched["ball_flight"]["events"]),
                            "track_joins": len(joins)})
            report("postprocess", (k + 1) / len(todo), f"post-processing segment {k + 1}/{len(todo)}")
        m.mark("postprocess", "done")
        check_cancel()

        index = {"pipeline_version": PIPELINE_VERSION, "source": {"name": video.name, **norm_meta["probe"]},
                 "normalization": {k: v for k, v in norm_meta.get("result", {}).items() if k != "source_time_s"},
                 "video": "normalized.mp4", "fps": fps, "frames": info["frames_decoded"],
                 "shots": shots, "segments": results, "annotated_video": None}

        # ── render ────────────────────────────────────────────────────────────
        if render:
            from .render import render_annotated
            ann = workdir / "results" / "annotated.mp4"
            if not m.done("render", ann):
                m.mark("render", "running")
                render_annotated(norm_path, [workdir / "results" / r["file"] for r in results], ann,
                                 on_progress=lambda f: report("render", f, "rendering annotated video"),
                                 cancel=is_cancelled)
                m.mark("render", "done")
            index["annotated_video"] = "results/annotated.mp4"
        report("render", 1.0, "done")

        _write_json(workdir / "results" / "analysis.json", index)
        m.data.update(status="completed", error=None)
        m.save()
        return index
    except AnalysisCancelled:
        m.data.update(status="cancelled")
        m.save()
        raise
    except AnalysisError as e:
        m.data.update(status="failed", error=e.as_dict())
        m.save()
        raise
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised for the caller
        m.data.update(status="failed", error={"code": "internal_error", "message": f"{type(e).__name__}: {e}"})
        m.save()
        raise
