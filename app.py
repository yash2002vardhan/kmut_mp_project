import numpy as np
import cv2
import gradio as gr
from pathlib import Path

from classifier import load_model, CLASSES
from feature_extraction import extract_features
from preprocessing import preprocess
from segmentation import segment_particles

model = load_model()


def _overlay_particles(bgr: np.ndarray, particle_mask: np.ndarray) -> np.ndarray:
    """Return RGB image with detected particles highlighted in red."""
    overlay = bgr.copy()
    overlay[particle_mask == 255] = [0, 0, 220]   # BGR red
    blended = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def classify(image: np.ndarray) -> tuple[str, dict, np.ndarray]:
    if image is None:
        return "No image provided", {}, None

    # Gradio passes RGB numpy array — save to tmp and run pipeline
    tmp = Path("/tmp/gradio_input.png")
    cv2.imwrite(str(tmp), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    bgr, circle_mask = preprocess(tmp)
    particle_mask = segment_particles(bgr, circle_mask)
    feats = extract_features(tmp).reshape(1, -1)

    pred_idx = model.predict(feats)[0]
    probs = model.predict_proba(feats)[0]
    predicted_class = CLASSES[pred_idx]
    prob_dict = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    overlay_img = _overlay_particles(bgr, particle_mask)
    return predicted_class, prob_dict, overlay_img


with gr.Blocks(title="Microplastics Classifier") as demo:
    gr.Markdown("# Microplastics Classifier")
    gr.Markdown("Upload a UV fluorescence image of microplastics to identify the polymer type.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            predicted_label = gr.Label(label="Predicted Class")
            prob_output = gr.Label(label="Class Probabilities", num_top_classes=4)
            overlay_output = gr.Image(label="Detected Particles (highlighted)")

    classify_btn.click(
        fn=classify,
        inputs=input_image,
        outputs=[predicted_label, prob_output, overlay_output],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
