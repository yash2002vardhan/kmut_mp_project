import tempfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

from detect import DEFAULT_CONF, resolve_weights
from measure import (Analysis, analyze, counts_by_polymer, find_particle_at,
                     render, summarize, to_rows, write_csv)

_WEIGHTS = resolve_weights()

PARTICLE_HEADERS = ["ID", "Polymer", "Conf", "Major (µm)", "Minor (µm)",
                    "ECD (µm)", "Area (µm²)", "Aspect", "Method"]
SUMMARY_HEADERS = ["Polymer", "Count", "Mean ECD (µm)", "Min ECD (µm)", "Max ECD (µm)"]


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _scale_note(cal) -> str:
    if cal.source == "fallback":
        return ("**Scale: fallback value used.** The filter paper circle could not be "
                f"found in this image, so sizes assume {cal.um_per_px:.2f} µm/pixel "
                "from previous images. Treat every dimension as approximate.")
    note = f"**Scale:** {cal.um_per_px:.2f} µm/pixel, measured from the 4.7 cm filter paper."
    if cal.clipped:
        note += ("  \n**Warning:** the filter circle runs past the image edge, so the "
                 "fitted radius — and every size below — may be wrong.")
    else:
        note += (f"  \nOne pixel is {cal.um_per_px:.0f} µm, so dimensions below "
                 f"~{cal.um_per_px * 5:.0f} µm rest on only a few pixels and carry "
                 "a large relative error.")
    return note


def _detail(p) -> str:
    return (f"**#{p.particle_id} — {p.polymer}** ({p.confidence:.0%} confidence)  \n"
            f"Major axis {p.major_axis_um:.0f} µm · minor axis {p.minor_axis_um:.0f} µm · "
            f"ECD {p.ecd_um:.0f} µm · area {p.area_um2:,.0f} µm² · "
            f"aspect {p.aspect_ratio:.2f} · measured from "
            f"{'the particle outline' if p.method == 'mask' else 'the detection box'}")


def analyze_image(image: np.ndarray, conf: float):
    if image is None:
        return {}, None, "", [], [], None, None, ""
    if not _WEIGHTS.exists():
        return ({"error: train first (python train.py --data data.yaml)": 1.0},
                None, "", [], [], None, None, "")

    # Gradio passes an RGB numpy array; Ultralytics/OpenCV work in BGR.
    analysis = analyze(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), conf=conf)
    particles = analysis.particles

    counts = dict(sorted(counts_by_polymer(particles).items(), key=lambda kv: -kv[1]))
    total = len(particles)
    label = {f"{cls} ({n})": n / total for cls, n in counts.items()} if total else {}

    particle_rows = [[p.particle_id, p.polymer, p.confidence, p.major_axis_um,
                      p.minor_axis_um, p.ecd_um, p.area_um2, p.aspect_ratio, p.method]
                     for p in particles]
    summary_rows = [[r["polymer"], r["count"], r["mean_ecd_um"],
                     r["min_ecd_um"], r["max_ecd_um"]] for r in summarize(particles)]

    csv_path = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
    write_csv(to_rows(particles), csv_path)

    hint = ("Click a particle in the image, or a row in the table, to isolate it."
            if total else "")
    return (label, _rgb(render(analysis)), _scale_note(analysis.calibration),
            summary_rows, particle_rows, str(csv_path), analysis, hint)


def select_row(analysis: Analysis | None, evt: gr.SelectData):
    """Highlight the particle whose table row was clicked."""
    if analysis is None or evt.index is None:
        return gr.skip(), gr.skip()
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not isinstance(row, int) or not 0 <= row < len(analysis.particles):
        return gr.skip(), gr.skip()
    p = analysis.particles[row]
    return _rgb(render(analysis, p.particle_id)), _detail(p)


def select_point(analysis: Analysis | None, evt: gr.SelectData):
    """Highlight the particle clicked in the image."""
    # index is [x, y] in original image pixels, or None when the click misses.
    if analysis is None or evt.index is None:
        return gr.skip(), gr.skip()
    x, y = int(evt.index[0]), int(evt.index[1])
    p = find_particle_at(analysis, x, y)
    if p is None:
        return _rgb(render(analysis)), "No particle there — showing all detections."
    return _rgb(render(analysis, p.particle_id)), _detail(p)


def show_all(analysis: Analysis | None):
    if analysis is None:
        return gr.skip(), gr.skip()
    return _rgb(render(analysis)), ""


with gr.Blocks(title="Microplastics Multi-Class Detector") as demo:
    gr.Markdown("# Microplastics Multi-Class Detector")
    gr.Markdown(
        "Upload a UV fluorescence image containing a mix of microplastics. "
        "The YOLO model detects and classifies each particle (PET / PP / PS / PVC) "
        "individually, then measures it in micrometres by calibrating against the "
        "4.7 cm filter paper."
    )

    state = gr.State()

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            conf_slider = gr.Slider(
                0.0, 1.0, value=DEFAULT_CONF, step=0.05,
                label="Confidence threshold",
            )
            detect_btn = gr.Button("Detect and measure", variant="primary")

        with gr.Column():
            counts_output = gr.Label(label="Particle counts by type")
            # Non-interactive so it renders as a static preview that emits click
            # coordinates, rather than as an upload widget.
            annotated_output = gr.Image(label="Detections", interactive=False)
            hint_output = gr.Markdown()
            detail_output = gr.Markdown()
            show_all_btn = gr.Button("Show all particles", size="sm")

    scale_output = gr.Markdown()

    gr.Markdown("### Size summary by polymer")
    summary_output = gr.Dataframe(headers=SUMMARY_HEADERS, interactive=False, wrap=True)

    gr.Markdown(
        "### Per-particle measurements\n"
        "`ECD` is the equivalent circular diameter (orientation-independent). "
        "`Method` is `mask` when the particle outline was resolved, or `bbox` when it "
        "was too faint to segment and the detection box was used instead, which "
        "overestimates elongated particles."
    )
    particles_output = gr.Dataframe(headers=PARTICLE_HEADERS, interactive=False, wrap=True)
    csv_output = gr.DownloadButton("Download CSV", variant="secondary")

    detect_btn.click(
        fn=analyze_image,
        inputs=[input_image, conf_slider],
        outputs=[counts_output, annotated_output, scale_output, summary_output,
                 particles_output, csv_output, state, hint_output],
    )
    particles_output.select(
        fn=select_row, inputs=state, outputs=[annotated_output, detail_output],
    )
    annotated_output.select(
        fn=select_point, inputs=state, outputs=[annotated_output, detail_output],
    )
    show_all_btn.click(
        fn=show_all, inputs=state, outputs=[annotated_output, detail_output],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
