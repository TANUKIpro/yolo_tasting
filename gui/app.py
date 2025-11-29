"""
Main Gradio application for YOLO Tasting GUI.

Provides an interactive interface for model selection, parameter tuning,
and real-time object detection visualization.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for the GUI. Install with: pip install gradio"
    )

from .model_registry import (
    get_versions, get_tasks_for_version, get_sizes_for_version_task,
    get_class_presets, get_preset_classes, get_model_key, get_model_config
)
from .detector_manager import get_detector_manager, DetectorManager


def format_detections(detections: List[Dict[str, Any]], task: str) -> str:
    """
    Format detection results for display.

    Args:
        detections: List of detection dictionaries
        task: Current task type

    Returns:
        Formatted string for display
    """
    if not detections:
        return "No detections"

    lines = []

    if task in ["Detection", "Detection (Custom Classes)", "OBB"]:
        lines.append(f"### Detected {len(detections)} object(s)\n")
        for i, det in enumerate(detections, 1):
            conf = det.get("confidence", 0)
            cls_name = det.get("class_name", "unknown")
            bbox = f"[{det.get('x_min', 0)}, {det.get('y_min', 0)}, {det.get('x_max', 0)}, {det.get('y_max', 0)}]"
            lines.append(f"{i}. **{cls_name}** ({conf:.2%}) - BBox: {bbox}")

    elif task == "Pose":
        lines.append(f"### Detected {len(detections)} pose(s)\n")
        for det in detections:
            person_id = det.get("person_id", 0)
            conf = det.get("confidence", 0)
            keypoints = det.get("keypoints", [])
            n_kpts = len(keypoints[0]) if keypoints else 0
            lines.append(f"- Person {person_id}: {n_kpts} keypoints (conf: {conf:.2%})")

    elif task == "Segmentation":
        lines.append(f"### Detected {len(detections)} segment(s)\n")
        for i, det in enumerate(detections, 1):
            cls_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0)
            mask_shape = det.get("mask_shape", "N/A")
            lines.append(f"{i}. **{cls_name}** ({conf:.2%}) - Mask: {mask_shape}")

    elif task == "Classification":
        lines.append(f"### Top {len(detections)} classification(s)\n")
        for i, det in enumerate(detections, 1):
            cls_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0)
            lines.append(f"{i}. **{cls_name}** ({conf:.2%})")

    return "\n".join(lines)


class YOLOTastingApp:
    """
    Gradio application for YOLO model testing and comparison.
    """

    def __init__(self):
        """Initialize the application."""
        self.manager = get_detector_manager()
        self._current_image: Optional[np.ndarray] = None
        self._current_task: str = "Detection"

    def on_version_change(self, version: str) -> Tuple[gr.Dropdown, gr.Dropdown, gr.Column]:
        """
        Handle YOLO version change.

        Returns updated task dropdown, size dropdown, and custom classes visibility.
        """
        tasks = get_tasks_for_version(version)
        default_task = tasks[0] if tasks else "Detection"

        sizes = get_sizes_for_version_task(version, default_task)
        default_size = "Medium" if "Medium" in sizes else sizes[0] if sizes else "Medium"

        # Show custom classes panel for YOLO-World
        show_classes = version == "YOLO-World"

        return (
            gr.Dropdown(choices=tasks, value=default_task, interactive=True),
            gr.Dropdown(choices=sizes, value=default_size, interactive=True),
            gr.Column(visible=show_classes)
        )

    def on_task_change(self, version: str, task: str) -> gr.Dropdown:
        """Handle task type change."""
        sizes = get_sizes_for_version_task(version, task)
        default_size = "Medium" if "Medium" in sizes else sizes[0] if sizes else "Medium"
        return gr.Dropdown(choices=sizes, value=default_size, interactive=True)

    def on_preset_change(self, preset: str) -> Tuple[str, gr.Textbox]:
        """Handle class preset change."""
        classes = get_preset_classes(preset)

        if classes is None:
            # COCO default
            return "", gr.Textbox(interactive=False, placeholder="Using default COCO classes")
        elif preset == "Custom":
            return "", gr.Textbox(interactive=True, placeholder="Enter classes separated by comma")
        else:
            classes_str = ", ".join(classes)
            return classes_str, gr.Textbox(interactive=False)

    def load_model(
        self,
        version: str,
        task: str,
        size: str,
        preset: str,
        custom_classes_text: str
    ) -> str:
        """
        Load the selected model.

        Returns status message.
        """
        # Determine custom classes
        custom_classes = None

        if version == "YOLO-World":
            if preset == "Custom" and custom_classes_text.strip():
                custom_classes = [c.strip() for c in custom_classes_text.split(",") if c.strip()]
            elif preset != "COCO (80 classes)":
                custom_classes = get_preset_classes(preset)

        # Store current task for detection formatting
        self._current_task = task

        # Load model
        success, message = self.manager.load_model(
            version=version,
            task=task,
            size=size,
            custom_classes=custom_classes
        )

        if success:
            info = self.manager.get_model_info()
            return f"**{info['display_name']}** loaded successfully\n\n{info['description']}"
        else:
            return f"Error: {message}"

    def update_params(
        self,
        conf: float,
        iou: float,
        img_size: int,
        max_det: int
    ) -> None:
        """Update detection parameters."""
        self.manager.update_params(
            conf_threshold=conf,
            iou_threshold=iou,
            img_size=img_size,
            max_det=max_det
        )

    def run_detection(
        self,
        image: Optional[np.ndarray],
        conf: float,
        iou: float,
        img_size: int,
        max_det: int
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Run detection on the input image.

        Returns annotated image and detection results text.
        """
        if image is None:
            return None, "Please upload an image"

        if not self.manager.is_loaded:
            return None, "Please load a model first"

        # Update parameters
        self.update_params(conf, iou, img_size, max_det)

        # Store image for re-detection
        self._current_image = image.copy()

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run detection
        detections, annotated, status = self.manager.detect(image_bgr)

        if annotated is None:
            return None, status

        # Convert BGR to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Format results
        results_text = format_detections(detections, self._current_task)

        return annotated_rgb, results_text

    def on_param_change(
        self,
        image: Optional[np.ndarray],
        conf: float,
        iou: float,
        img_size: int,
        max_det: int
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Handle parameter change - re-run detection if image exists.
        """
        if image is None or not self.manager.is_loaded:
            return None, "Upload image and load model"

        return self.run_detection(image, conf, iou, img_size, max_det)


def create_app() -> gr.Blocks:
    """
    Create the Gradio application.

    Returns:
        Gradio Blocks application
    """
    app_instance = YOLOTastingApp()

    with gr.Blocks(title="YOLO Tasting GUI") as app:
        gr.Markdown("""
        # YOLO Tasting GUI

        Interactive object detection with YOLOv5, YOLOv11, and YOLO-World models.
        Select a model, adjust parameters, and upload an image to see detection results.
        """)

        with gr.Row():
            # Left column - Model Selection
            with gr.Column(scale=1):
                gr.Markdown("## Model Selection")

                version_dropdown = gr.Dropdown(
                    choices=get_versions(),
                    value="YOLOv11",
                    label="YOLO Version",
                    interactive=True
                )

                task_dropdown = gr.Dropdown(
                    choices=get_tasks_for_version("YOLOv11"),
                    value="Detection",
                    label="Task Type",
                    interactive=True
                )

                size_dropdown = gr.Dropdown(
                    choices=get_sizes_for_version_task("YOLOv11", "Detection"),
                    value="Medium",
                    label="Model Size",
                    interactive=True
                )

                # Custom classes panel (for YOLO-World)
                with gr.Column(visible=False) as custom_classes_panel:
                    gr.Markdown("### Custom Classes (YOLO-World)")

                    preset_dropdown = gr.Dropdown(
                        choices=get_class_presets(),
                        value="RoboCup@Home",
                        label="Class Preset",
                        interactive=True
                    )

                    custom_classes_text = gr.Textbox(
                        label="Classes",
                        placeholder="Classes will appear here...",
                        lines=3,
                        interactive=False
                    )

                load_btn = gr.Button("Load Model", variant="primary")

                model_status = gr.Markdown(
                    "No model loaded",
                    elem_classes=["model-status"]
                )

                gr.Markdown("---")
                gr.Markdown("## Detection Parameters")

                conf_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    interactive=True
                )

                iou_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="IoU Threshold (NMS)",
                    interactive=True
                )

                imgsize_dropdown = gr.Dropdown(
                    choices=[320, 416, 512, 640, 832, 1024, 1280],
                    value=640,
                    label="Image Size",
                    interactive=True
                )

                maxdet_slider = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=300,
                    step=10,
                    label="Max Detections",
                    interactive=True
                )

            # Right column - Image I/O
            with gr.Column(scale=2):
                gr.Markdown("## Detection")

                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Input Image",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            interactive=True
                        )

                    with gr.Column():
                        output_image = gr.Image(
                            label="Detection Result",
                            type="numpy",
                            interactive=False
                        )

                detect_btn = gr.Button("Run Detection", variant="primary", size="lg")

                gr.Markdown("## Results")
                results_text = gr.Markdown(
                    "Detection results will appear here...",
                    elem_classes=["results-box"]
                )

        # Event handlers

        # Version change -> update task and size dropdowns
        version_dropdown.change(
            fn=app_instance.on_version_change,
            inputs=[version_dropdown],
            outputs=[task_dropdown, size_dropdown, custom_classes_panel]
        )

        # Task change -> update size dropdown
        task_dropdown.change(
            fn=app_instance.on_task_change,
            inputs=[version_dropdown, task_dropdown],
            outputs=[size_dropdown]
        )

        # Preset change -> update classes text
        preset_dropdown.change(
            fn=app_instance.on_preset_change,
            inputs=[preset_dropdown],
            outputs=[custom_classes_text, custom_classes_text]
        )

        # Load model button
        load_btn.click(
            fn=app_instance.load_model,
            inputs=[
                version_dropdown,
                task_dropdown,
                size_dropdown,
                preset_dropdown,
                custom_classes_text
            ],
            outputs=[model_status]
        )

        # Detection button
        detect_btn.click(
            fn=app_instance.run_detection,
            inputs=[
                input_image,
                conf_slider,
                iou_slider,
                imgsize_dropdown,
                maxdet_slider
            ],
            outputs=[output_image, results_text]
        )

        # Parameter changes -> re-run detection (live update)
        for param_component in [conf_slider, iou_slider]:
            param_component.release(
                fn=app_instance.on_param_change,
                inputs=[
                    input_image,
                    conf_slider,
                    iou_slider,
                    imgsize_dropdown,
                    maxdet_slider
                ],
                outputs=[output_image, results_text]
            )

        # Image size and max det changes
        for param_component in [imgsize_dropdown, maxdet_slider]:
            param_component.change(
                fn=app_instance.on_param_change,
                inputs=[
                    input_image,
                    conf_slider,
                    iou_slider,
                    imgsize_dropdown,
                    maxdet_slider
                ],
                outputs=[output_image, results_text]
            )

        # Image upload -> auto detect if model loaded
        input_image.change(
            fn=app_instance.run_detection,
            inputs=[
                input_image,
                conf_slider,
                iou_slider,
                imgsize_dropdown,
                maxdet_slider
            ],
            outputs=[output_image, results_text]
        )

    return app


def launch_app(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    debug: bool = False
) -> None:
    """
    Launch the Gradio application.

    Args:
        share: Create a public shareable link
        server_name: Server hostname
        server_port: Server port
        debug: Enable debug mode
    """
    app = create_app()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        debug=debug
    )


if __name__ == "__main__":
    launch_app(debug=True)
