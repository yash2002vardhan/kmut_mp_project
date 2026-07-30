"""
Physical size measurement for detected microplastic particles.

YOLO reports pixel-space boxes. This module turns them into physical dimensions:

1. Calibrate scale from the filter paper visible in the image. Its diameter is
   known (4.7 cm), so the fitted circle gives micrometres per pixel.
2. Refine each box down to the actual particle pixels before measuring. An
   axis-aligned box inflates the size of any elongated particle lying at an
   angle, so axes are taken from a rotated-rectangle fit on the particle mask.

Each particle's outline is kept alongside its measurements so a caller can
re-render a highlight for one particle without re-running inference.

Usage:
    python measure.py --image /path/to/img.jpeg --out particles.csv
    python measure.py --dir /path/to/images --out particles.csv
"""

import argparse
import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from detect import DEFAULT_CONF, DEFAULT_IMGSZ, predict
from logger_config import get_logger

logger = get_logger(__name__)

FILTER_DIAMETER_MM = 47.0  # 4.7 cm filter paper — the calibration reference

# Measured on the Roboflow-resampled 1350x1350 images: the fitted filter radius
# lands at 588-619 px, i.e. 47000 um over ~1180 px. Used only when the circle
# cannot be found, in which case the scale is no longer image-specific.
FALLBACK_UM_PER_PX = 39.8

# A rotated-rectangle fit on fewer pixels than this is noise, not a measurement;
# such particles fall back to their YOLO box.
MIN_MASK_AREA_PX = 6

# Below this local contrast a box holds no resolvable particle, so Otsu would be
# thresholding noise. Calibrated on the test split: boxes placed on empty
# background scored 8 at the 90th percentile, while every one of the 596 real
# detections scored 16.4 or above.
MIN_BOX_CONTRAST = 14.0

# Padding ring around a box, used to sample the local background. Scaled to the
# box so small particles get a proportionate ring, and bounded so a large box
# does not reach into its neighbours.
_PAD_FRACTION, _PAD_MIN_PX, _PAD_MAX_PX = 0.25, 3, 12
_MIN_RING_PX = 20

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Outline colours (BGR), picked to stand off both the blue background and the
# pale green/white particles rather than to match polymer fluorescence.
POLYMER_COLOURS = {
    "PET": (255, 0, 255),    # magenta
    "PP": (0, 165, 255),     # orange
    "PS": (0, 0, 255),       # red
    "PVC": (0, 255, 255),    # yellow
}
_FALLBACK_COLOUR = (255, 255, 255)
_SELECTED_COLOUR = (0, 255, 0)
_DIMMED_COLOUR = (110, 110, 110)
_DIM_FACTOR = 0.35
_SPOTLIGHT_PAD_PX = 8

# Particles are only tens of pixels across, so accept a near miss when a click
# lands outside every box.
_CLICK_REACH_PX = 40.0


@dataclass
class Calibration:
    """Image-to-physical scale, plus where it came from."""

    um_per_px: float
    source: str  # "filter_circle" | "fallback"
    circle: tuple[int, int, int] | None  # (cx, cy, r) in px
    clipped: bool  # filter circle runs past the image edge -> scale suspect

    @property
    def reliable(self) -> bool:
        return self.source == "filter_circle" and not self.clipped


@dataclass
class Particle:
    """One measured particle. Pixel columns are kept alongside the physical ones
    so a reader can see how many pixels each measurement rests on."""

    particle_id: int
    polymer: str
    confidence: float
    center_x_px: int
    center_y_px: int
    box_w_px: int
    box_h_px: int
    major_axis_px: float
    minor_axis_px: float
    area_px: float
    major_axis_um: float
    minor_axis_um: float
    ecd_um: float
    area_um2: float
    aspect_ratio: float
    method: str  # "mask" (refined) | "bbox" (mask fit failed)


PARTICLE_FIELDS = [f.name for f in fields(Particle)]


@dataclass
class Analysis:
    """Everything one image yielded, kept together so the UI can re-render a
    highlight without re-running inference."""

    particles: list[Particle]
    outlines: dict[int, np.ndarray]  # particle_id -> contour in image coords
    calibration: Calibration
    image: np.ndarray  # BGR original, the base for every render


def _find_filter_circle(bgr: np.ndarray) -> tuple[int, int, int] | None:
    """Fit the circular filter paper. Same Hough parameters as the classical
    pipeline's dish detection, which was tuned on these images."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=50,
        param2=30,
        minRadius=min(h, w) // 4,
        maxRadius=min(h, w) // 2,
    )
    if circles is None:
        return None
    cx, cy, r = np.round(circles[0, 0]).astype(int)
    return int(cx), int(cy), int(r)


def calibrate(bgr: np.ndarray) -> Calibration:
    """Derive micrometres-per-pixel from the filter paper in this image.

    Calibrating per image matters: the training images split into two sessions
    whose fitted radii differ by ~5%, so a single hard-coded constant would carry
    that as a systematic scale error.
    """
    circle = _find_filter_circle(bgr)
    if circle is None:
        logger.warning(
            f"Filter circle not found; using fallback scale {FALLBACK_UM_PER_PX} um/px. "
            "Sizes are only as good as that assumption."
        )
        return Calibration(FALLBACK_UM_PER_PX, "fallback", None, False)

    cx, cy, r = circle
    h, w = bgr.shape[:2]
    clipped = cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h
    if clipped:
        logger.warning(
            "Filter circle extends past the image edge; the fitted radius, and so "
            "every size below, may be wrong."
        )

    um_per_px = FILTER_DIAMETER_MM * 1000.0 / (2.0 * r)
    logger.info(f"Scale: {um_per_px:.2f} um/px (filter radius {r} px)")
    return Calibration(um_per_px, "filter_circle", circle, clipped)


def _refine_box(bgr: np.ndarray, bx1: int, by1: int,
                bx2: int, by2: int) -> tuple[np.ndarray, int] | None:
    """Segment the particle inside one detection box.

    Thresholds adaptively: the padding ring around the box supplies the local
    background colour, and the particle is whatever deviates from it. Fixed HSV
    cut-offs were tried first and do not transfer between imaging sessions — the
    constants tuned for the classical pipeline missed 97% of PVC on this dataset,
    because these particles read as a subtle brightening of the blue rather than
    a hue shift. Local contrast has no such calibration to get wrong.

    Returns (blob mask, area in px) clipped to the box, or None when the box is
    too flat to hold a resolvable particle.
    """
    h, w = bgr.shape[:2]
    box_w, box_h = bx2 - bx1, by2 - by1
    pad = int(np.clip(round(_PAD_FRACTION * min(box_w, box_h)),
                      _PAD_MIN_PX, _PAD_MAX_PX))
    px1, py1 = max(0, bx1 - pad), max(0, by1 - pad)
    px2, py2 = min(w, bx2 + pad), min(h, by2 + pad)

    roi = bgr[py1:py2, px1:px2]
    if roi.size == 0:
        return None

    # Lab puts hue shift, brightening and desaturation on one comparable scale,
    # so a single distance threshold covers all four polymers.
    lab = cv2.cvtColor(cv2.GaussianBlur(roi, (3, 3), 0),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    ox1, oy1 = bx1 - px1, by1 - py1
    ring = np.ones(lab.shape[:2], dtype=bool)
    ring[oy1:oy1 + box_h, ox1:ox1 + box_w] = False
    if int(ring.sum()) < _MIN_RING_PX:
        return None

    dist = np.linalg.norm(lab - np.median(lab[ring], axis=0), axis=2)
    inner = dist[oy1:oy1 + box_h, ox1:ox1 + box_w]
    if inner.size == 0:
        return None

    # Otsu always splits, even on noise, so gate on contrast before trusting it.
    contrast = float(np.percentile(inner, 99))
    if contrast < MIN_BOX_CONTRAST:
        return None

    scaled = np.clip(inner / contrast * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return _target_blob(mask)


def _target_blob(mask: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Isolate the particle the box is centred on.

    Prefers the component covering the crop centre — in dense images a
    neighbouring particle can intrude at the crop edge and outweigh the target.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None

    h, w = mask.shape[:2]
    idx = int(labels[h // 2, w // 2])
    if idx == 0:  # centre landed on background
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    return (labels == idx).astype(np.uint8) * 255, int(stats[idx, cv2.CC_STAT_AREA])


def _largest_contour(blob: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _axes_from_contour(contour: np.ndarray) -> tuple[float, float]:
    """Major and minor axis in px from a rotated-rectangle fit."""
    (_, _), (a, b), _ = cv2.minAreaRect(contour)
    # A detected blob is at least one pixel across; a zero-width fit is a
    # degenerate single-pixel-wide line, not a zero-thickness particle.
    major, minor = max(a, b), min(a, b)
    return max(major, 1.0), max(minor, 1.0)


def measure_boxes(bgr: np.ndarray, result,
                  cal: Calibration) -> tuple[list[Particle], dict[int, np.ndarray]]:
    """Measure every detection in a YOLO result against a calibrated scale.

    Returns the measurements plus each particle's outline in whole-image
    coordinates, so the UI can highlight the particle rather than its box.
    """
    h, w = bgr.shape[:2]
    names = result.names
    scale = cal.um_per_px
    particles: list[Particle] = []
    outlines: dict[int, np.ndarray] = {}

    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

        # Work in whole pixels, and report the box dimensions actually measured
        # so a mask can never appear larger than the box that contains it.
        bx1, by1 = max(0, int(round(x1))), max(0, int(round(y1)))
        bx2, by2 = min(w, int(round(x2))), min(h, int(round(y2)))
        box_w, box_h = bx2 - bx1, by2 - by1
        if box_w <= 0 or box_h <= 0:
            logger.warning(f"Skipping degenerate box {i}: {box_w}x{box_h} px")
            continue

        found = _refine_box(bgr, bx1, by1, bx2, by2)
        contour = _largest_contour(found[0]) if found else None

        if found and contour is not None and found[1] >= MIN_MASK_AREA_PX:
            major_px, minor_px = _axes_from_contour(contour)
            area_px = float(found[1])
            method = "mask"
            outlines[i] = contour + np.array([bx1, by1], dtype=contour.dtype)
        else:
            # No usable mask: fall back to the box. Area uses the inscribed
            # ellipse rather than the full rectangle, since a particle does not
            # fill its bounding box.
            major_px, minor_px = max(box_w, box_h), min(box_w, box_h)
            area_px = math.pi / 4.0 * box_w * box_h
            method = "bbox"
            outlines[i] = np.array(
                [[[bx1, by1]], [[bx2, by1]], [[bx2, by2]], [[bx1, by2]]], dtype=np.int32
            )

        particles.append(
            Particle(
                particle_id=i,
                polymer=names[int(box.cls.item())],
                confidence=round(float(box.conf.item()), 4),
                center_x_px=int(round((x1 + x2) / 2)),
                center_y_px=int(round((y1 + y2) / 2)),
                box_w_px=int(round(box_w)),
                box_h_px=int(round(box_h)),
                major_axis_px=round(major_px, 1),
                minor_axis_px=round(minor_px, 1),
                area_px=round(area_px, 1),
                major_axis_um=round(major_px * scale, 1),
                minor_axis_um=round(minor_px * scale, 1),
                ecd_um=round(2.0 * math.sqrt(area_px / math.pi) * scale, 1),
                area_um2=round(area_px * scale * scale, 1),
                aspect_ratio=round(major_px / minor_px, 2),
                method=method,
            )
        )

    return particles, outlines


def analyze(
    image: str | Path | np.ndarray,
    weights: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
) -> "Analysis":
    """Detect and measure every particle in one image."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image}")
    else:
        bgr = image

    result = predict(bgr, weights=weights, conf=conf, imgsz=imgsz)
    cal = calibrate(bgr)
    particles, outlines = measure_boxes(bgr, result, cal)
    return Analysis(particles=particles, outlines=outlines, calibration=cal, image=bgr)


def find_particle_at(analysis: "Analysis", x: int, y: int) -> Particle | None:
    """The particle a click at (x, y) refers to.

    Prefers the smallest box containing the point, so clicking inside a small
    particle that overlaps a large one selects the small one. Falls back to the
    nearest centre within a short reach, since particles here are only tens of
    pixels across and an exact hit is fiddly.
    """
    hits = [p for p in analysis.particles
            if abs(x - p.center_x_px) <= p.box_w_px / 2
            and abs(y - p.center_y_px) <= p.box_h_px / 2]
    if hits:
        return min(hits, key=lambda p: p.box_w_px * p.box_h_px)

    if not analysis.particles:
        return None
    nearest = min(analysis.particles,
                  key=lambda p: (x - p.center_x_px) ** 2 + (y - p.center_y_px) ** 2)
    within = math.hypot(x - nearest.center_x_px, y - nearest.center_y_px)
    return nearest if within <= _CLICK_REACH_PX else None


def render(analysis: "Analysis", selected_id: int | None = None) -> np.ndarray:
    """Draw detections on the image (BGR).

    With no selection every particle is outlined and numbered. With a selection
    the rest of the image is dimmed and only that particle is drawn at full
    brightness, ringed so it stays findable at full-image zoom.
    """
    bgr = analysis.image
    selected = next((p for p in analysis.particles
                     if p.particle_id == selected_id), None) if selected_id else None

    if selected is None:
        out = bgr.copy()
        for p in analysis.particles:
            colour = POLYMER_COLOURS.get(p.polymer, _FALLBACK_COLOUR)
            cv2.drawContours(out, [analysis.outlines[p.particle_id]], -1, colour, 1)
            _label(out, str(p.particle_id),
                   (p.center_x_px + p.box_w_px // 2 + 3, p.center_y_px), 0.4)
        return out

    out = (bgr * _DIM_FACTOR).astype(np.uint8)
    for p in analysis.particles:
        if p.particle_id != selected.particle_id:
            cv2.drawContours(out, [analysis.outlines[p.particle_id]], -1, _DIMMED_COLOUR, 1)

    # Restore true pixels around the selection so its colour can still be judged.
    # Circular, so the bright region ends exactly where the ring is drawn rather
    # than showing a rectangular seam through it.
    cx, cy = selected.center_x_px, selected.center_y_px
    radius = int(max(selected.box_w_px, selected.box_h_px) / 2) + _SPOTLIGHT_PAD_PX + 6
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2 = min(bgr.shape[1], cx + radius + 1)
    y2 = min(bgr.shape[0], cy + radius + 1)
    disc = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.circle(disc, (cx - x1, cy - y1), radius, 255, -1)
    inside = disc > 0
    out[y1:y2, x1:x2][inside] = bgr[y1:y2, x1:x2][inside]

    cv2.drawContours(out, [analysis.outlines[selected.particle_id]], -1, _SELECTED_COLOUR, 2)
    cv2.circle(out, (cx, cy), radius, _SELECTED_COLOUR, 1, cv2.LINE_AA)
    _label(out, f"#{selected.particle_id} {selected.polymer} {selected.ecd_um:.0f}um",
           (x1, max(14, cy - radius - 6)), 0.5)
    return out


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    """Text with a dark outline so it reads over both bright particles and background."""
    for colour, thickness in ((0, 0, 0), 3), ((255, 255, 255), 1):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thickness, cv2.LINE_AA)


def counts_by_polymer(particles: list[Particle]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in particles:
        counts[p.polymer] = counts.get(p.polymer, 0) + 1
    return counts


def summarize(particles: list[Particle]) -> list[dict]:
    """Per-polymer size summary, sorted by count. ECD is the summary dimension
    because it is orientation-independent."""
    rows = []
    for polymer in sorted(counts_by_polymer(particles)):
        ecd = [p.ecd_um for p in particles if p.polymer == polymer]
        rows.append({
            "polymer": polymer,
            "count": len(ecd),
            "mean_ecd_um": round(sum(ecd) / len(ecd), 1),
            "min_ecd_um": round(min(ecd), 1),
            "max_ecd_um": round(max(ecd), 1),
        })
    return sorted(rows, key=lambda r: -r["count"])


def to_rows(particles: list[Particle], image_name: str | None = None) -> list[dict]:
    """Flatten particles to CSV rows, prefixing an image column when batching."""
    rows = []
    for p in particles:
        row = asdict(p)
        rows.append({"image": image_name, **row} if image_name is not None else row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    fieldnames = list(rows[0].keys()) if rows else PARTICLE_FIELDS
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        logger.warning(f"No particles detected; wrote header-only CSV to {path}")
    else:
        logger.info(f"Wrote {len(rows)} particles to {path}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure detected microplastic particles")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Single image to measure")
    src.add_argument("--dir", help="Folder of images to measure")
    p.add_argument("--out", default="particles.csv", help="CSV output path")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--weights", default=None, help="Override weights path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image:
        paths = [Path(args.image)]
    else:
        folder = Path(args.dir)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")

    all_rows: list[dict] = []
    for path in paths:
        a = analyze(path, weights=args.weights, conf=args.conf, imgsz=args.imgsz)
        all_rows.extend(to_rows(a.particles, image_name=path.name))
        flag = "" if a.calibration.reliable else "  [scale unreliable]"
        logger.info(f"{path.name}: {len(a.particles)} particles, "
                    f"{a.calibration.um_per_px:.2f} um/px{flag}")

    write_csv(all_rows, args.out)
    for row in summarize([Particle(**{k: v for k, v in r.items() if k != "image"})
                          for r in all_rows]):
        logger.info(f"{row['polymer']}: n={row['count']}  "
                    f"ECD mean {row['mean_ecd_um']} um "
                    f"(range {row['min_ecd_um']}-{row['max_ecd_um']})")
