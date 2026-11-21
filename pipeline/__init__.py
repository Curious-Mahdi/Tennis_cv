"""Reusable Tennis Vision analysis pipeline. Entry point: pipeline.run.run_analysis."""
import os as _os

# Ultralytics prints one line per frame unless told otherwise; set before it is imported
# (pipeline.export imports trackers, which imports ultralytics).
_os.environ.setdefault("YOLO_VERBOSE", "False")
