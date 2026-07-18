"""
YOLO training for per-particle microplastics detection (multi-class).

One YOLO model localizes AND classifies every particle in a (possibly mixed)
image — this replaces the classical HSV-segmentation + SVM pipeline that lives
in the project root.

Dataset: export the Roboflow project as "YOLOv11" (Object Detection). It
produces train/valid(/test) image+label subdirs and a data.yaml. Point --data
at a data.yaml whose train/val/test entries locate the image folders (see the
data.yaml in this folder for the current absolute-path layout).

Usage:
    python multiclass_detection/train.py --data multiclass_detection/data.yaml
    python multiclass_detection/train.py --data ... --model yolo11m.pt --imgsz 1280
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from logger_config import get_logger

logger = get_logger(__name__)

# Ultralytics writes runs here; best weights land in RUN_DIR/weights/best.pt
PROJECT_DIR = Path(__file__).parent / "runs"
RUN_NAME = "microplastics"


def _default_device() -> str:
    # Ultralytics does not auto-select Apple's GPU; pick it explicitly.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def train(
    data: str,
    model: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 16,
    device: str | None = None,
) -> Path:
    device = device or _default_device()
    yolo = YOLO(model)
    logger.info(f"Training {model} on {data} (imgsz={imgsz}, epochs={epochs}, device={device})")

    yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,          # high res: microplastic particles are small objects
        batch=batch,
        device=device,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        patience=25,
        # Polymer identity is encoded in fluorescence colour (PP=yellow-green,
        # PS=cyan-green, PET=white glow). Disable all colour augmentation so it
        # does not scramble the class signal. Geometric augmentation stays on.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )

    best = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    logger.info(f"Training complete. Best weights: {best}")

    metrics = yolo.val()
    logger.info(f"Validation mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO microplastics detector")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolo11s.pt",
                   help="Base weights: yolo11n/s/m/l/x.pt (n=fastest, x=most accurate)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None,
                   help="mps / cpu / 0 (CUDA). Default: auto (mps on Apple Silicon)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")
    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device)
