import cv2
import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)

# Background characterisation (measured empirically on UV fluorescence images):
#   H ≈ 100–125  (blue hue in OpenCV 0–180 scale)
#   S ≈ 180–255  (highly saturated blue)
#   V ≈ 160–220  (medium-bright, not clipped)
#
# Particle signatures:
#   PET  → same blue H, but S < 150 (desaturated = whitish glow) AND V > 220
#   PP   → H ≈ 25–50  (yellow-green fluorescence)
#   PS   → H ≈ 45–70  (cyan-green fluorescence)
#   PVC  → H ≈ 30–65, very low contrast but slight greenish tint

_BG_H_LO, _BG_H_HI = 100, 128   # blue hue range
_BG_S_MIN = 170                   # background is strongly saturated


def _particle_mask_raw(hsv: np.ndarray) -> np.ndarray:
    """
    Detect particle pixels using three complementary rules:
    1. Hue shift: pixel hue outside the background blue band (catches PP, PS, PVC)
    2. Saturation drop: S < 150 with high brightness (catches white-glow PET)
    3. Brightness spike: V > 230 (catches any very bright particle regardless of hue)
    """
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)

    hue_shift = (h < _BG_H_LO) | (h > _BG_H_HI)          # non-blue hue
    sat_drop = (s < 150) & (v > 200)                        # desaturated + bright (PET)
    bright_spike = v > 230                                   # any very bright pixel

    return ((hue_shift | sat_drop | bright_spike)).astype(np.uint8) * 255


def segment_particles(bgr: np.ndarray, circle_mask: np.ndarray) -> np.ndarray:
    """
    Return binary mask (uint8, 0/255) of fluorescent particle pixels.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw = _particle_mask_raw(hsv)

    # Apply circle mask
    particles = cv2.bitwise_and(raw, circle_mask)

    # Light morphological cleanup — only remove obvious salt noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    particles = cv2.morphologyEx(particles, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove specks smaller than 8 px²
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(particles, connectivity=8)
    clean = np.zeros_like(particles)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            clean[labels == i] = 255

    return clean


def get_particle_stats(particle_mask: np.ndarray) -> dict:
    """Return basic morphology stats about detected particle blobs."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(particle_mask, connectivity=8)
    if num_labels <= 1:
        return {
            "count": 0,
            "mean_area": 0.0,
            "std_area": 0.0,
            "total_area": 0,
            "mean_aspect_ratio": 1.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    aspect_ratios = np.where(heights > 0, widths / heights, 1.0)

    return {
        "count": len(areas),
        "mean_area": float(np.mean(areas)),
        "std_area": float(np.std(areas)),
        "total_area": int(np.sum(areas)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
    }
