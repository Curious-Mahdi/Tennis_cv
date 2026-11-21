# Web viewer

Static HTML, CSS and JavaScript with no build step.

```bash
python web/serve.py          # http://localhost:8000
```

- `data/demo_frames.json` is written by `scripts/prepare_web_demo.py` and contains
  positions only.
- `media/` holds the clip cut and stills. They are broadcast-derived and gitignored, so
  regenerate them locally. Without them the page opens in frame-stepping mode.

Use `serve.py` rather than `python -m http.server`. On the development machine the
built-in server intermittently truncated responses, and `serve.py` also answers the
byte-range requests that video seeking needs.

## What is drawn, and from which field

| On screen | Field in `demo_frames.json` |
|---|---|
| Player markers and trails | `players.N.court.xy_m_smoothed` (untick *Smoothed* for `xy_m`). Hidden unless `court.status == "measured"`; trails break at gaps. |
| Ball ring and trail on the video | `ball.clean`: solid = `detected`, dashed = `interpolated`. The trail joins consecutive frames of the same `tracklet` only. |
| Red ✕ "rejected detection" | `ball.clean.status == "rejected"`, at `ball.image.xy` |
| Ball on the court | `ball.flight` (`xy_m`, `height_m`, `uncertainty_m`, `residual_px`). Solid on a measured bounce; dashed and translucent otherwise, with an uncertainty ring above 0.35 m. |
| ✕ / ◇ court marks, timeline events | `ball_flight.events` (`status`: measured / estimated / unavailable) |
| Court occupancy, charts, table | `analytics.N` |

## Sync

The frame shown is `floor(mediaTime × fps)` of the frame the video presents
(`requestVideoFrameCallback`), never a separate timer.

## Keys

- Space: play or pause.
- ← / →: step one frame.
- Esc: back to the clip overview.
- `?frame=106` opens the page at that frame.
