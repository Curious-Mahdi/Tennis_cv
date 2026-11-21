"""
Regression tests: the refactored pipeline must reproduce what the original scripts produced
on the demo footage.

Goldens in tests/data/ were written by scripts/export_web_data.py and
scripts/prepare_web_demo.py BEFORE the pipeline package existed, from the 150-frame
detection checkpoint of input_videos/tennis_clip.mp4:
    demo150_detections.pkl.gz          raw detections (players, keypoints, ball)
    demo150_export_golden.json.gz      export_web_data.py output
    demo150_enriched_golden.json.gz    prepare_web_demo.py output

The source video is broadcast footage and is not distributed, so these tests skip when it
is absent. No model inference runs here.
"""
import copy
import gzip
import json
import pickle
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
VIDEO = ROOT / "input_videos" / "tennis_clip.mp4"
needs_video = pytest.mark.skipif(not VIDEO.exists(), reason="demo footage not present (not redistributed)")

# Fields that legitimately differ between runs or by documented design.
VOLATILE = [("generated_at",), ("analysis", "wall_clock_s_this_invocation"),
            ("analysis", "segment", "reason"), ("source", "video")]


def load_gz(name, binary=False):
    with gzip.open(DATA / name, "rb") as f:
        return pickle.load(f) if binary else json.loads(f.read().decode("utf-8"))


def strip_volatile(d):
    d = copy.deepcopy(d)
    for path in VOLATILE:
        node = d
        for k in path[:-1]:
            node = node.get(k, {})
        node.pop(path[-1], None)
    return d


def first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
    elif a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


@pytest.fixture(scope="module")
def demo_state():
    return load_gz("demo150_detections.pkl.gz", binary=True)["state"]


@needs_video
def test_build_export_matches_original_script(demo_state):
    from pipeline.config import load_config
    from pipeline.export import build_export, video_info

    info = video_info(VIDEO)
    export, rows = build_export(str(VIDEO), load_config(), info, 0, 150, demo_state, segment_reason="golden")
    export = json.loads(json.dumps(export, allow_nan=False))
    golden = load_gz("demo150_export_golden.json.gz")
    assert export["source"]["video"] == "tennis_clip.mp4"         # name only, by design
    diff = first_difference(strip_volatile(golden), strip_volatile(export))
    assert diff is None, diff
    assert len(rows) == info["frames_decoded"]


@pytest.mark.slow
@needs_video
def test_detection_with_reused_models_matches_original_script(demo_state, tmp_path):
    """Real inference on the first frames: same detections as the original script, twice
    in a row on ONE ModelBundle (so the ByteTrack reset between runs is proven)."""
    import numpy as np

    from pipeline.export import detect_segment
    from pipeline.models import ModelBundle

    models = ModelBundle()
    n = 8
    runs = [detect_segment(VIDEO, 0, n, tmp_path / f"run{k}.pkl", models) for k in range(2)]
    for st in runs:
        assert st["players"] == demo_state["players"][:n]
        assert all(np.allclose(a, b) for a, b in zip(st["keypoints"], demo_state["keypoints"][:n]))
        # the last frame's ball is clamped at the segment edge, so compare the others
        assert st["ball"][: n - 1] == demo_state["ball"][: n - 1]


def test_enrich_matches_original_script():
    """Ball clean-up, flight fit and movement analytics, from the golden export alone."""
    from pipeline.enrich import enrich

    out, seg = enrich(load_gz("demo150_export_golden.json.gz"), log=lambda *_: None)
    out["frames"] = seg
    out = json.loads(json.dumps(out, allow_nan=False))      # compare what is written to disk
    golden = load_gz("demo150_enriched_golden.json.gz")
    golden.pop("demo")                      # media metadata (clip cut), tested separately
    diff = first_difference(strip_volatile(golden), strip_volatile(out))
    assert diff is None, diff
