import cv2
import numpy as np
from pathlib import Path
from logger_config import get_logger

logger = get_logger(__name__)

TARGET_SIZE = (512, 512)


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def detect_circle_mask(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Detect the circular petri dish and return a binary mask + (cx, cy, r).
    Falls back to a centered circle covering 85% of the shorter dimension.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2, # Minimum distance between detected circles
        param1=50, # Gradient threshold for edge detection
        param2=30, # Accumulator threshold for circle detection
        minRadius=min(h, w) // 4,
        maxRadius=min(h, w) // 2, # Maximum radius of circles to detect
    )

    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
    else:
        cx, cy = w // 2, h // 2
        r = int(min(h, w) * 0.42) # 0.42 is just a tried and tested value based on the images in the dataset
        logger.debug("HoughCircles failed, using fallback circle.")

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask, (cx, cy, r)


def crop_and_resize(img: np.ndarray, circle_mask: np.ndarray, params: tuple[int, int, int]) -> np.ndarray:
    """Apply circle mask (zero-out corners) and resize to TARGET_SIZE."""
    masked = img.copy()
    masked[circle_mask == 0] = 0 # outside the petri dish the img will be black, only circular region remains visible
    cx, cy, r = params
    h, w = img.shape[:2]
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    cropped = masked[y1:y2, x1:x2]
    return cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def preprocess(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.
    Returns (preprocessed_bgr, circle_mask_512x512).

    CLAHE is intentionally omitted: in these UV fluorescence images the background
    is already medium-bright blue, and histogram equalisation inflates it further,
    making particle/background separation harder.
    """
    img = load_image(path)
    circle_mask, params = detect_circle_mask(img)
    cropped = crop_and_resize(img, circle_mask, params)

    h, w = cropped.shape[:2]
    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(new_mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)

    return cropped, new_mask
