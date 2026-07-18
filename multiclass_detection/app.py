import cv2
import numpy as np
import gradio as gr

from detect import detect, resolve_weights

_WEIGHTS = resolve_weights()


def classify(image: np.ndarray, conf: float) -> tuple[dict, np.ndarray]:
    if image is None:
        return {}, None
    if not _WEIGHTS.exists():
        return {"error: train first (python multiclass_detection/train.py --data ...)": 1.0}, None

    # Gradio passes an RGB numpy array; Ultralytics/OpenCV work in BGR.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    counts, total, annotated = detect(bgr, conf=conf)

    counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    label = {f"{cls} ({n})": n / total for cls, n in counts.items()} if total else {}
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return label, annotated_rgb


with gr.Blocks(title="Microplastics Multi-Class Detector") as demo:
    gr.Markdown("# Microplastics Multi-Class Detector")
    gr.Markdown(
        "Upload a UV fluorescence image containing a mix of microplastics. "
        "The YOLO model detects and classifies each particle (PET / PP / PS / PVC) "
        "individually and reports a count per polymer type."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            conf_slider = gr.Slider(
                0.0, 1.0, value=0.25, step=0.05,
                label="Confidence threshold",
            )
            detect_btn = gr.Button("Detect particles", variant="primary")

        with gr.Column():
            counts_output = gr.Label(label="Particle counts by type")
            annotated_output = gr.Image(label="Detections")

    detect_btn.click(
        fn=classify,
        inputs=[input_image, conf_slider],
        outputs=[counts_output, annotated_output],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
