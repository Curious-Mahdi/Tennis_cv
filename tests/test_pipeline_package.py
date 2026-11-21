"""
Tests for the pipeline package with real orchestration code and a fake model bundle:
validation, normalisation, segmentation, track stitching, run_analysis (progress,
isolation, cancellation, resume, model reuse) and rendering. No weights, no footage.
"""
import json
from pathlib import Path

import av
import pytest

from pipeline import video as vid
from pipeline.errors import AnalysisCancelled, AnalysisError
from pipeline.run import run_analysis
from pipeline.segments import court_segments, find_shots
from pipeline.tracks import stitch_track_fragments

from .pipeline_fakes import FakeModels, number_frames, write_video


def _probe(**kw):
    base = {"container": ".mp4", "duration_s": 60.0, "rotation": 0, "width": 1280, "height": 720,
            "fps": 29.97, "codec": "h264", "pix_fmt": "yuv420p", "constant_frame_rate": True}
    base.update(kw)
    return base


# ── validation ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,code", [
    ({"duration_s": 601.0}, "too_long"),
    ({"container": ".avi"}, "unsupported_container"),
    ({"rotation": 90}, "rotated_video"),
    ({"width": 720, "height": 1280}, "unsupported_aspect"),
    ({"width": 640, "height": 360}, "resolution_too_low"),
    ({"fps": 15.0}, "unsupported_frame_rate"),
    ({"fps": 40.0}, "unsupported_frame_rate"),
])
def test_plan_rejects_unsupported_input(kw, code):
    with pytest.raises(AnalysisError) as e:
        vid.plan(_probe(**kw))
    assert e.value.code == code


def test_plan_frame_rate_classes_never_duplicate_frames():
    assert vid.plan(_probe(fps=25.0))["keep_every"] == 1          # stays 25: inside the measured band
    assert vid.plan(_probe(fps=59.94))["keep_every"] == 2
    assert vid.plan(_probe(fps=120.0))["keep_every"] == 4
    assert vid.plan(_probe())["passthrough"] is True
    assert vid.plan(_probe(width=1920, height=1080))["passthrough"] is False
    assert vid.plan(_probe(duration_s=600.0))["keep_every"] == 1   # exactly 10 minutes is allowed


def test_normalize_60fps_1080p(tmp_path):
    src = tmp_path / "in.mp4"
    write_video(src, 60, fps=60, width=1920, height=1080)
    info = vid.probe(src)
    plan = vid.plan(info)
    assert plan["keep_every"] == 2 and plan["resize"] and not plan["passthrough"]
    res = vid.normalize(src, tmp_path / "out.mp4", plan)
    c = av.open(str(tmp_path / "out.mp4"))
    s = c.streams.video[0]
    frames = list(c.decode(s))
    assert (s.codec_context.width, s.codec_context.height) == (1280, 720)
    assert len(frames) == res["frames"] == 30
    assert abs(float(s.average_rate) - 30.0) < 0.01
    assert all(abs(f.time - i / 30.0) < 1e-3 for i, f in enumerate(frames))
    c.close()
    assert res["source_time_s"][:3] == [0.0, round(2 / 60, 4), round(4 / 60, 4)]   # every 2nd source frame


def test_passthrough_keeps_bytes(tmp_path):
    src = tmp_path / "in.mp4"
    write_video(src, 30)
    plan = vid.plan(vid.probe(src))
    assert plan["passthrough"]
    vid.normalize(src, tmp_path / "out.mp4", plan)
    assert (tmp_path / "out.mp4").read_bytes() == src.read_bytes()


# ── segments ──────────────────────────────────────────────────────────────────

def test_cut_detected_and_only_court_shot_analysable(tmp_path):
    src = tmp_path / "in.mp4"
    write_video(src, 120, cut_at=90)
    shots, cuts = find_shots(src)
    assert [c["frame"] for c in cuts] == [90] and shots == [(0, 90), (90, 120)]
    scored = court_segments(src, shots, FakeModels(), 30.0)
    assert scored[0]["analysable"] and scored[0]["line_support"] > 0.5
    assert not scored[1]["analysable"]                       # 1 s and no court


# ── track stitching ───────────────────────────────────────────────────────────

def _box(x, y, h=60):
    return [x - 15, y - h, x + 15, y]


def test_stitch_joins_a_short_dropout():
    dets = [{3: _box(600, 160)} for _ in range(10)] + [{} for _ in range(6)] + [{10: _box(604, 161)} for _ in range(10)]
    stitched, joins = stitch_track_fragments(dets)
    assert len(joins) == 1 and joins[0]["gap_frames"] == 6
    assert {tid for fr in stitched for tid in fr} == {3}


def _fragments(second, gap=2):
    """Track 3 for 10 frames, then `gap` empty frames, then track 10 for 5 frames."""
    return [{3: _box(600, 160)} for _ in range(10)] + [{} for _ in range(gap)] + [{10: second} for _ in range(5)]


def test_stitch_refuses_overlapping_tracks():
    dets = [{3: _box(600, 160)} for _ in range(10)]
    dets[-1] = {3: _box(600, 160), 10: _box(604, 161)}       # both exist on one frame: two people
    dets += [{10: _box(604, 161)} for _ in range(5)]
    assert stitch_track_fragments(dets)[1] == []


@pytest.mark.parametrize("second,gap", [
    (_box(900, 161), 2),          # too far away for the gap
    (_box(604, 161, h=130), 2),   # different size
    (_box(604, 161), 20),         # gap longer than 0.5 s
])
def test_stitch_refuses_implausible_joins(second, gap):
    assert stitch_track_fragments(_fragments(second, gap))[1] == []


# ── run_analysis ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("clip") / "match.mp4"
    write_video(p, 120, cut_at=90, frame_fn=number_frames)
    return p


def _strip(d):
    d = json.loads(json.dumps(d))
    d.pop("generated_at", None)
    d["analysis"].pop("detector_seconds", None)
    d["analysis"].pop("resumed_at_frames", None)
    d["players"].pop("track_stitching", None)
    return d


def test_run_analysis_end_to_end(tmp_path, clip, monkeypatch):
    work = tmp_path / "job"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    events = []
    models = FakeModels()
    index = run_analysis(clip, work, on_progress=events.append, models=models)

    assert Path.cwd() == outside and list(outside.iterdir()) == []          # nothing written outside work/
    prog = [e["progress"] for e in events]
    assert prog == sorted(prog) and prog[-1] == pytest.approx(1.0)
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert [s["start"] for s in index["segments"]] == [0] and index["segments"][0]["end"] == 90
    seg = json.loads((work / "results" / index["segments"][0]["file"]).read_text())
    assert seg["players"]["track_stitching"]["joins"][0]["gap_frames"] == 6      # far-player dropout re-joined
    assert seg["players"]["selection"]["player_2_coverage"] > 0.9
    c = av.open(str(work / "results" / "annotated.mp4"))
    assert sum(1 for _ in c.decode(c.streams.video[0])) == 120
    c.close()


def test_cancel_then_resume_matches_uninterrupted_run(tmp_path, clip):
    ref = run_analysis(clip, tmp_path / "ref", models=FakeModels(), render=False)
    ref_seg = json.loads((tmp_path / "ref" / "results" / ref["segments"][0]["file"]).read_text())

    work = tmp_path / "job"
    seen = []

    def progress(e):
        seen.append(e)

    def cancel():
        return any(e["stage"] == "detect" and e["stage_progress"] > 0.45 for e in seen)

    with pytest.raises(AnalysisCancelled):
        run_analysis(clip, work, on_progress=progress, cancel=cancel, models=FakeModels(), render=False)
    assert json.loads((work / "manifest.json").read_text())["status"] == "cancelled"
    assert list((work / "checkpoints").glob("seg_*.pkl"))

    models = FakeModels()
    index = run_analysis(clip, work, models=models, render=False)
    seg = json.loads((work / "results" / index["segments"][0]["file"]).read_text())
    assert seg["analysis"]["resumed_at_frames"]                          # it really resumed
    assert models.court_calls < 90                                       # and did not redo finished frames
    assert _strip(seg) == _strip(ref_seg)


def test_models_are_reused_across_analyses(tmp_path, clip):
    models = FakeModels()
    run_analysis(clip, tmp_path / "a", models=models, render=False)
    run_analysis(clip, tmp_path / "b", models=models, render=False)
    assert models.loads == 1 and models.resets >= 2                     # one load, a reset per segment


def test_unanalysable_video_fails_with_a_code(tmp_path):
    src = tmp_path / "graphic.mp4"
    write_video(src, 90, cut_at=0)                                       # no court at all
    with pytest.raises(AnalysisError) as e:
        run_analysis(src, tmp_path / "job", models=FakeModels())
    assert e.value.code == "no_court_segment"
    assert json.loads((tmp_path / "job" / "manifest.json").read_text())["error"]["code"] == "no_court_segment"
