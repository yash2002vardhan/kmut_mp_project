"""
YOLO inference for per-particle microplastics detection (multi-class).

Loads trained weights and reports, for one image, how many particles of each
polymer type were detected (plus an annotated image with boxes).
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from logger_config import get_logger

logger = get_logger(__name__)

RUNS_DIR = Path(__file__).parent / "runs"
# Canonical location; train.py may write to microplastics-2, -3, ... on reruns.
DEFAULT_WEIGHTS = RUNS_DIR / "microplastics" / "weights" / "best.pt"

# 0.25 is the F1-optimal operating point measured on the test split.
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 1024  # high res: microplastic particles are small objects

_model_cache: dict[str, YOLO] = {}


def resolve_weights(weights: str | Path | None = None) -> Path:
    """Resolve which weights file to use. If none given, prefer the canonical
    run, else the most recently trained microplastics* run."""
    if weights is not None:
        return Path(weights)
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    runs = sorted(RUNS_DIR.glob("microplastics*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else DEFAULT_WEIGHTS


def load_detector(weights: str | Path | None = None) -> YOLO:
    """Load (and cache) a YOLO model from weights (latest run if unspecified)."""
    path = resolve_weights(weights)
    key = str(path)
    if key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path}. Train first with train.py."
            )
        _model_cache[key] = YOLO(key)
        logger.info(f"Loaded YOLO detector from {path}")
    return _model_cache[key]


def predict(
    image: str | Path | np.ndarray,
    weights: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
):
    """Run the detector on one image and return the raw Ultralytics Result."""
    model = load_detector(weights)
    return model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]


def detect(
    image: str | Path | np.ndarray,
    weights: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
) -> tuple[dict, int, np.ndarray]:
    """
    Detect and classify particles in one image.

    Returns:
        counts:    {class_name: n} for classes with at least one detection
        total:     total number of detected particles
        annotated: BGR image with boxes + labels drawn (as from Ultralytics plot())
    """
    result = predict(image, weights=weights, conf=conf, imgsz=imgsz)

    names = result.names
    counts: dict[str, int] = {}
    for cls_id in result.boxes.cls.tolist():
        name = names[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1

    total = int(len(result.boxes))
    annotated = result.plot()  # BGR numpy array
    return counts, total, annotated
