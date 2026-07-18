import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from logger_config import get_logger

logger = get_logger(__name__)

CLASSES = ["PET", "PP", "PS", "PVC"]
MODEL_PATH = Path(__file__).parent / "model.pkl"


def build_svm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=50.0, gamma="scale", probability=True, random_state=42)),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def train(X_train: np.ndarray, y_train: np.ndarray, model_name: str = "svm") -> Pipeline:
    builders = {"svm": build_svm, "rf": build_random_forest}
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(builders)}")
    model = builders[model_name]()
    logger.info(f"Training {model_name.upper()} on {len(y_train)} samples...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray, split_name: str = "Test") -> dict:
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=CLASSES)
    cm = confusion_matrix(y, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"{split_name} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"{'='*50}\n")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "predictions": y_pred}


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model


def predict(image_path: str | Path, model: Pipeline | None = None) -> tuple[str, dict]:
    """Predict the class of a single image. Returns (class_name, probabilities_dict)."""
    from feature_extraction import extract_features

    if model is None:
        model = load_model()

    feats = extract_features(image_path).reshape(1, -1)
    pred_idx = model.predict(feats)[0]
    probs = model.predict_proba(feats)[0]
    return CLASSES[pred_idx], {cls: float(p) for cls, p in zip(CLASSES, probs)}
