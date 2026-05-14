import cv2
import numpy as np
from pathlib import Path

from preprocessing import preprocess
from segmentation import segment_particles, get_particle_stats
from logger_config import get_logger

logger = get_logger(__name__)

H_BINS, S_BINS, V_BINS = 18, 8, 8
MIN_CONTOUR_AREA = 8


# ---------------------------------------------------------------------------
# Feature group 1: HSV histograms of segmented particle pixels (34)
# ---------------------------------------------------------------------------

def _hsv_channel_histograms(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixel_mask = mask == 255
    if not np.any(pixel_mask):
        return np.zeros(H_BINS + S_BINS + V_BINS, dtype=np.float32)

    def norm(x):
        s = x.sum()
        return (x / s).astype(np.float32) if s > 0 else x.astype(np.float32)

    h_hist, _ = np.histogram(hsv[:, :, 0][pixel_mask], bins=H_BINS, range=(0, 180))
    s_hist, _ = np.histogram(hsv[:, :, 1][pixel_mask], bins=S_BINS, range=(0, 256))
    v_hist, _ = np.histogram(hsv[:, :, 2][pixel_mask], bins=V_BINS, range=(0, 256))
    return np.concatenate([norm(h_hist), norm(s_hist), norm(v_hist)])


# ---------------------------------------------------------------------------
# Feature group 2: colour statistics of particle pixels (15)
# ---------------------------------------------------------------------------

def _particle_color_stats(bgr: np.ndarray, hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixel_mask = mask == 255
    if not np.any(pixel_mask):
        return np.zeros(15, dtype=np.float32)

    b = bgr[:, :, 0][pixel_mask].astype(np.float64)
    g = bgr[:, :, 1][pixel_mask].astype(np.float64)
    r = bgr[:, :, 2][pixel_mask].astype(np.float64)
    hh = hsv[:, :, 0][pixel_mask].astype(np.float64)
    ss = hsv[:, :, 1][pixel_mask].astype(np.float64)
    vv = hsv[:, :, 2][pixel_mask].astype(np.float64)

    eps = 1e-6
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)

    return np.array([
        b_mean, np.std(b),
        g_mean, np.std(g),
        r_mean, np.std(r),
        np.mean(hh), np.std(hh),
        np.mean(ss), np.std(ss),
        np.mean(vv), np.std(vv),
        g_mean / (b_mean + eps),   # G/B ratio ; + eps is done to avoid division by zero cases
        r_mean / (b_mean + eps),   # R/B ratio
        g_mean / (r_mean + eps),   # G/R ratio
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature group 3: global circle statistics (17)
# ---------------------------------------------------------------------------

def _global_circle_stats(hsv: np.ndarray, circle_mask: np.ndarray) -> np.ndarray:
    px = circle_mask > 0
    v = hsv[:, :, 2][px].astype(np.float64)
    h = hsv[:, :, 0][px].astype(np.float64)
    s = hsv[:, :, 1][px].astype(np.float64)

    n_total = float(len(v))
    n_v200 = np.sum(v > 200) / n_total
    n_v215 = np.sum(v > 215) / n_total
    n_v230 = np.sum(v > 230) / n_total
    n_nonblue = np.sum((hsv[:, :, 0][px] < 100) | (hsv[:, :, 0][px] > 128)) / n_total
    v_p75, v_p90, v_p95, v_p99 = (np.percentile(v, q) for q in (75, 90, 95, 99))

    cand = px & ((hsv[:, :, 1] < 150) | (hsv[:, :, 2] > 215))
    if np.any(cand):
        h_cand = hsv[:, :, 0][cand].astype(np.float64)
        h_cand_mean, h_cand_std = float(np.mean(h_cand)), float(np.std(h_cand))
    else:
        h_cand_mean, h_cand_std = 0.0, 0.0

    n_whiteglow = float(np.sum((hsv[:, :, 1][px] < 150) & (v > 215))) / n_total

    return np.array([
        n_v200, n_v215, n_v230,
        n_nonblue,
        v_p75, v_p90, v_p95, v_p99,
        h_cand_mean, h_cand_std,
        n_whiteglow,
        np.mean(v), np.std(v),
        np.mean(h), np.std(h),
        np.mean(s),
        float(n_total),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature group 4: per-particle morphology aggregation (22)
# ---------------------------------------------------------------------------

def _per_particle_morphology(particle_mask: np.ndarray, circle_mask: np.ndarray) -> np.ndarray:
    zero = np.zeros(22, dtype=np.float32)
    contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if len(contours) == 0:
        return zero

    areas, perims, circularities, aspects, solidities, extents = [], [], [], [], [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        areas.append(area)
        perims.append(perim)

        circ = 4 * np.pi * area / (perim ** 2)
        circularities.append(min(circ, 1.0))

        _, _, bw, bh = cv2.boundingRect(cnt)
        asp = float(bw) / (bh + 1e-6)
        aspects.append(max(asp, 1 / (asp + 1e-6)))

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidities.append(area / (hull_area + 1e-6))

        extents.append(area / (bw * bh + 1e-6))

    def stats5(arr):
        a = np.array(arr, dtype=np.float64)
        if len(a) == 0:
            return [0.0] * 5
        return [np.mean(a), np.std(a), np.median(a), np.percentile(a, 25), np.percentile(a, 75)]

    circle_area = float(np.sum(circle_mask > 0))
    total_area = float(sum(areas))

    return np.array(
        stats5(circularities)   # 5
        + stats5(aspects)       # 5
        + stats5(solidities)    # 5
        + [
            float(len(contours)),
            total_area,
            total_area / circle_area if circle_area > 0 else 0.0,
            float(np.mean(areas)),
            float(np.std(areas)),
            float(np.mean(perims)),
            float(np.mean(extents)),
        ],                      # 7
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(path: str | Path) -> np.ndarray:
    """
    Full feature extraction for one image.
    Feature vector layout (88 dims):
      HSV_hists(34) | particle_color(15) | global_circle(17) | per_particle_morph(22)
    """
    bgr, circle_mask = preprocess(path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    particle_mask = segment_particles(bgr, circle_mask)

    return np.concatenate([
        _hsv_channel_histograms(hsv, particle_mask),
        _particle_color_stats(bgr, hsv, particle_mask),
        _global_circle_stats(hsv, circle_mask),
        _per_particle_morphology(particle_mask, circle_mask),
    ])


def extract_dataset_features(
    split_dir: str | Path,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features for all images in a split. Returns X (N, 88), y, paths."""
    split_dir = Path(split_dir)
    X_list, y_list, paths = [], [], []

    for label, cls in enumerate(classes):
        cls_dir = split_dir / cls
        images = sorted(
            p for p in cls_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        logger.info(f"Extracting {len(images)} images from {cls_dir.name}/{cls}")
        for img_path in images:
            try:
                feats = extract_features(img_path)
                X_list.append(feats)
                y_list.append(label)
                paths.append(str(img_path))
            except Exception as e:
                logger.warning(f"Skipping {img_path.name}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), paths
