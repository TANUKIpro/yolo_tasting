"""
YOLOv5 detector implementation.
"""
from typing import Dict, List, Tuple, Union, Any

import torch
import numpy as np
import cv2

# Patch torch.load for YOLOv5 compatibility
_torch_load_original = torch.load.__wrapped__ if hasattr(torch.load, "__wrapped__") else torch.load


def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _torch_load_original(*args, **kwargs)


torch.load = _safe_torch_load

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import colors, Annotator
from yolov5.utils.torch_utils import select_device

from .base import BaseDetector


class YOLOv5Detector(BaseDetector):
    """
    YOLOv5 object detector implementation.
    """

    def __init__(
        self,
        weights: str,
        imgsz: Union[int, List[int]] = 640,
        conf_thres: float = 0.5,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        device: str = "",
        half: bool = False
    ):
        """
        Initialize YOLOv5 detector.

        Args:
            weights: Path to weights file (.pt)
            imgsz: Inference size (int or [width, height])
            conf_thres: Confidence threshold
            iou_thres: NMS IoU threshold
            max_det: Maximum detections
            device: Device to use ('', '0', 'cpu')
            half: Use FP16 inference
        """
        # Handle imgsz as int or list
        if isinstance(imgsz, int):
            imgsz_val = imgsz
        else:
            imgsz_val = imgsz[0] if len(imgsz) > 0 else 640

        super().__init__(conf_thres, iou_thres, imgsz_val, max_det)

        self.device = select_device(device)
        self.half = half and self.device.type != "cpu"
        self._imgsz_list = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]

        # Load model
        print(f"Loading model: {weights}")
        self.model = attempt_load(weights, device=self.device)
        self.stride = int(self.model.stride.max())
        self._names = self.model.module.names if hasattr(self.model, "module") else self.model.names

        if self.half:
            self.model.half()

        # Check image size
        self._imgsz_list[0] = check_img_size(self._imgsz_list[0], s=self.stride)

        self._print_model_info(weights)

    def _print_model_info(self, weights: str) -> None:
        """Print model information."""
        print("Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Stride: {self.stride}")
        print(f"  Image size: {self._imgsz_list}")
        names_list = list(self._names.values()) if isinstance(self._names, dict) else self._names
        print(f"  Classes ({len(names_list)}): {names_list}")

    @property
    def names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        if isinstance(self._names, dict):
            return self._names
        return {i: name for i, name in enumerate(self._names)}

    @torch.no_grad()
    def detect(
        self,
        img_input: Union[str, np.ndarray],
        augment: bool = False
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run object detection.

        Args:
            img_input: Input image (numpy array BGR or file path)
            augment: Use test time augmentation

        Returns:
            Tuple of (detections list, annotated image)
        """
        # Load image
        if isinstance(img_input, str):
            img0 = cv2.imread(img_input)
        else:
            img0 = img_input.copy()

        # Preprocess (letterbox)
        img = letterbox(img0, self._imgsz_list, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Convert to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

        # Inference
        pred = self.model(img, augment=augment, visualize=False)[0]

        # NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres,
            None, False, max_det=self.max_det
        )

        # Process results
        detections = []
        annotator = Annotator(img0.copy(), line_width=2, example=str(self._names))

        for det in pred:
            if len(det):
                # Scale boxes
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    class_name = self._names[c] if isinstance(self._names, list) else self._names.get(c, str(c))
                    label = f"{class_name} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    detections.append({
                        "x_min": int(xyxy[0]),
                        "y_min": int(xyxy[1]),
                        "x_max": int(xyxy[2]),
                        "y_max": int(xyxy[3]),
                        "confidence": float(conf),
                        "class_name": class_name
                    })

        return detections, annotator.result()
