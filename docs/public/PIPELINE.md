# Analysis pipeline (`pipeline/`)

The detection, tracking, court and analytics code packaged behind one function, for use by
a worker process. The models, thresholds and calculations are the existing ones. The code
was moved, not rewritten, and regression tests compare it with goldens produced by the
original scripts.

```python
from pipeline.models import ModelBundle
from pipeline.run import run_analysis

models = ModelBundle()          # once per worker process: YOLOv8x, ResNet-50 court, TrackNet

index = run_analysis(
    "upload.mp4", "jobs/1234",  # input video, isolated work directory
    on_progress=print,          # {"stage", "stage_progress", "progress", "message"}
    cancel=stop_event,          # threading.Event or callable; checked between frames
    models=models,
)
```

`run_analysis` never changes the process working directory and writes only inside the
work directory. It raises `AnalysisError` with a stable `.code` for input it cannot
analyse, and `AnalysisCancelled` when cancelled. Re-running the same call after a crash or
a cancellation resumes the job.

## Stages

| Stage | What it does | Resumes from |
|---|---|---|
| probe | Container, codec, size, frame rate, duration, rotation, constant-frame-rate check; validates the input | `normalize.json` |
| normalize | H.264 MP4, 1280×720, 30-fps class, no audio; frame *i* at *i*/fps | `normalized.mp4` |
| segments | Camera cuts (colour-histogram distance), then a court check per shot using the existing line-support gate | `segments.json` |
| detect | Streaming YOLOv8x + ByteTrack, ResNet-50 keypoints and TrackNet, one frame in memory at a time; checkpoint every 25 frames | `checkpoints/seg_A_B.pkl`, mid-segment |
| postprocess | Track stitching, then the original post-processing (selection, gates, events, homography), then ball clean-up, flight fit and movement analytics | the stage outputs |
| render | Annotated H.264 MP4, streamed | `results/annotated.mp4` |

## Work directory

```
manifest.json              status, stage timings, error {code, message}
normalize.json             probe facts, plan, source time of every analysed frame
normalized.mp4             the analysed video (also the web player's clip)
segments.json              cuts and shots, each with its verdict and reason
checkpoints/seg_A_B.pkl    raw detections per segment
results/analysis.json      index: source, normalisation, shots, segment files
results/segment_A_B.json   per-segment result, in the web viewer's format
results/segment_A_B.csv    flat per-frame table
results/annotated.mp4      overlay video
```

## Input rules

| Rule | Value | Why |
|---|---|---|
| Duration | ≤ 10 minutes | Product limit |
| Container | MP4 or MOV | |
| Orientation | 16:9 landscape, no rotation flag | The court model only knows the broadcast-style view |
| Resolution | ≥ 960×540, scaled to 1280×720 | Pixel thresholds (ball speed bound, shot proximity) were set at 1280×720 |
| Frame rate | 23–31 fps kept 1:1; 46–62 fps every 2nd frame; 92–125 fps every 4th; otherwise rejected | Event thresholds count frames and were measured at 23–31 fps. Frames are never duplicated, because a duplicate looks like the ball stopping, so 25 fps stays 25. |

A file that is already H.264, 1280×720, 23–31 fps, constant frame rate and unrotated is
used byte-for-byte, so its results are identical to the original scripts'.

## Behaviour changes, and why

1. **Player track stitching** (`pipeline/tracks.py`). ByteTrack gives a player a new ID
   after a short detection dropout, and selection keeps one ID per player, so the frames
   under the other ID were discarded. Fragments are now re-joined when:
   - the second starts after the first ends;
   - the gap is ≤ 15 frames;
   - the feet are within ½ box height + 6 px per gap frame;
   - the box heights are within 1.5×.

   Every join is reported in `players.track_stitching`.

   On the demo, one join (track 3 → 10, 6-frame gap, 2.4 px) raises far-player coverage
   from 84.7% to 92.0%. The longest gap drops from 17 to 6 frames, and selection moves from
   `degraded` to `ok`. Events are unchanged, and the six undetected frames stay missing.
   The wrappers in `scripts/` do not stitch, so their output stays identical to before.
2. **Automatic segments** replace the manual `--end-frame`. On the demo, the shot pass
   finds exactly the two real cuts (frames 2335 and 2387) and selects [0, 2335), the
   segment previously chosen by hand.
3. `source.video` in results holds the file name rather than the path given, so a
   server's storage layout never appears in results.

## Tests

| File | Covers | Needs |
|---|---|---|
| `tests/test_pipeline_equivalence.py` | Moved post-processing and enrichment reproduce the original scripts' goldens; real detection on 8 frames matches, twice on one reused bundle | Demo footage; the detection test also needs weights (`slow`) |
| `tests/test_pipeline_package.py` | Validation codes, 60→30 fps normalisation, byte-identical passthrough, cut detection, stitching, end-to-end run (progress, isolation, render), cancel→resume equals an uninterrupted run, model reuse, failure codes | Nothing: a synthetic court video and a fake model bundle |

## Limits

- **One camera setup.** Only fixed, elevated, behind-the-baseline footage is supported. A
  handheld or panning camera can trigger false cuts, and other angles fail the court check
  and are refused with a reason.
- **Players are labelled by court side within each segment** (near / far). There is no
  re-identification across camera cuts, or when players change ends.
- **Resume splits tracks.** After resuming mid-segment, track IDs are offset. Stitching
  re-joins fragments that meet its rule, but tracks are not otherwise carried across the
  resume point.
- **Runtime.** Detection dominates the run time. See `scripts/benchmark_pipeline.py` and
  `notebooks/colab_benchmark_t4.ipynb` for measurements.
