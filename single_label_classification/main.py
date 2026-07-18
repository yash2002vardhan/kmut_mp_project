"""
Microplastics classification pipeline.

Usage:
    Train (uses Train+Validation, evaluates on all three splits):
        python main.py --mode train --model svm
        python main.py --mode train --model rf

    Predict a single image:
        python main.py --mode predict --image /path/to/image.png
        python main.py --mode predict --model rf --image /path/to/image.png
"""

import argparse
from pathlib import Path

import numpy as np

from feature_extraction import extract_dataset_features
from classifier import CLASSES, train, evaluate, save_model, load_model, predict
from logger_config import get_logger

logger = get_logger(__name__)

DATA_ROOT = Path("/Users/yashvardhan/Downloads/HSV Dataset")
TRAIN_DIR = DATA_ROOT / "Train"
VAL_DIR = DATA_ROOT / "Validation"
TEST_DIR = DATA_ROOT / "Test"


def run_train(model_name: str) -> None:
    logger.info("=== Feature extraction: Train ===")
    X_train, y_train, _ = extract_dataset_features(TRAIN_DIR, CLASSES)

    logger.info("=== Feature extraction: Validation ===")
    X_val, y_val, _ = extract_dataset_features(VAL_DIR, CLASSES)

    logger.info("=== Feature extraction: Test ===")
    X_test, y_test, _ = extract_dataset_features(TEST_DIR, CLASSES)

    # Evaluate on each split individually using train-only model
    model_train = train(X_train, y_train, model_name=model_name)
    evaluate(model_train, X_train, y_train, split_name="Train")
    evaluate(model_train, X_val, y_val, split_name="Validation")
    evaluate(model_train, X_test, y_test, split_name="Test")

    # Final model trained on train+val combined
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    logger.info("=== Training final model on Train+Validation ===")
    model_final = train(X_all, y_all, model_name=model_name)
    evaluate(model_final, X_test, y_test, split_name="Test (final model)")
    save_model(model_final)


def run_predict(image_path: str, model_name: str) -> None:
    model = load_model()
    cls, probs = predict(image_path, model)
    print(f"\nPredicted class: {cls}")
    print("Class probabilities:")
    for c, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {c}: {p:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microplastics classifier")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--model", choices=["svm", "rf"], default="svm")
    parser.add_argument("--image", type=str, help="Image path (predict mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        run_train(args.model)
    elif args.mode == "predict":
        if not args.image:
            raise ValueError("--image required for predict mode")
        run_predict(args.image, args.model)
