"""
pipeline/config.py
──────────────────
Configuration for the reusable pipeline, independent of the process working directory.

main.load_config resolves configs/config.yaml and every model path relative to the current
directory, which is why scripts/export_web_data.py used to os.chdir() at import time. A
worker that runs many analyses cannot change its working directory per job, so here the
same defaults and the same YAML merge (main._DEFAULTS, main._merge_yaml_into) are applied
with paths anchored at the repository root.

The configured values themselves (thresholds, model choices) are untouched.
"""
from __future__ import annotations

import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def _resolve_model_path(value: str) -> str:
    """Anchor a configured model path at the repository root.

    The player model is configured by name ("yolov8x"); Ultralytics appends ".pt" and
    downloads it if absent. Anchoring it at the root makes that download land in the
    repository (gitignored) instead of wherever the worker happens to run.
    """
    p = Path(value)
    if p.is_absolute():
        return str(p)
    if not p.suffix:
        p = p.with_suffix(".pt")
    return str(REPO_ROOT / p)


def load_config(path: str | Path | None = None) -> dict:
    """Built-in defaults <- configs/config.yaml <- `path` (optional), with absolute model paths.

    `cfg["model_labels"]` keeps the configured (relative) names for reporting, so results
    never contain the machine's filesystem layout.
    """
    import main  # the pipeline's existing defaults and merge rule

    cfg = copy.deepcopy(main._DEFAULTS)
    if BASE_CONFIG.exists():
        main._merge_yaml_into(cfg, str(BASE_CONFIG))
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.resolve() != BASE_CONFIG.resolve():
            if not p.exists():
                raise FileNotFoundError(f"config file not found: {p}")
            main._merge_yaml_into(cfg, str(p))
    cfg["model_labels"] = dict(cfg["models"])
    cfg["models"] = {k: _resolve_model_path(v) for k, v in cfg["models"].items()}
    return cfg


def model_label(cfg: dict, key: str) -> str:
    """The configured name of a model, as reported in outputs."""
    return cfg.get("model_labels", {}).get(key, cfg["models"][key])
