"""
YOLOv11 (Ultralytics) detector implementation.
"""
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import cv2

from ultralytics import YOLO

from .base import BaseDetector


class YOLOv11Detector(BaseDetector):
    """
    YOLOv11 / YOLO-World unified detector using Ultralytics.
    Supports object detection, pose estimation, and segmentation.
    """

    def __init__(
        self,
        model: Union[str, YOLO],
        conf_thres: float = 0.5,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        custom_classes: Optional[List[str]] = None
    ):
        """
        Initialize YOLOv11 detector.

        Args:
            model: Model name (e.g., 'yolo11m.pt') or YOLO instance
            conf_thres: Confidence threshold
            iou_thres: NMS IoU threshold
            imgsz: Inference size
            max_det: Maximum detections
            custom_classes: Custom classes for YOLO-World (None for standard COCO)
        """
        super().__init__(conf_thres, iou_thres, imgsz, max_det)

        # Load model
        if isinstance(model, str):
            self.model = YOLO(model)
            print(f"Model loaded: {model}")
        else:
            self.model = model

        # Set custom classes for YOLO-World
        if custom_classes is not None:
            self.set_classes(custom_classes)

        print(f"Task: {self.model.task}")
        print(f"Classes: {len(self.model.names)} classes")

    @property
    def names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        return self.model.names

    def set_classes(self, classes: List[str]) -> None:
        """
        Set custom classes for YOLO-World.

        Args:
            classes: List of class names to detect
        """
        # Handle device mismatch by moving to CPU temporarily
        try:
            original_device = next(self.model.model.parameters()).device
            self.model.to("cpu")
            self.model.set_classes(classes)
            self.model.to(original_device)
        except (StopIteration, AttributeError):
            # Fallback for models without parameters
            self.model.set_classes(classes)

        print(f"Classes set ({len(classes)}): {classes[:5]}{'...' if len(classes) > 5 else ''}")

    def detect(
        self,
        img_input: Union[str, np.ndarray],
        save: bool = False,
        save_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run object detection.

        Args:
            img_input: Input image (numpy array or file path)
            save: Whether to save result
            save_path: Path to save result

        Returns:
            Tuple of (detections list, annotated image)
        """
        # Run inference
        results = self.model(
            img_input,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False
        )

        # Process results
        result = results[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                detections.append({
                    "x_min": int(xyxy[0]),
                    "y_min": int(xyxy[1]),
                    "x_max": int(xyxy[2]),
                    "y_max": int(xyxy[3]),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        # Get annotated image
        annotated_img = result.plot()

        # Save if requested
        if save and save_path:
            cv2.imwrite(save_path, annotated_img)

        return detections, annotated_img

    def detect_pose(
        self,
        img_input: Union[str, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run pose estimation (requires pose model).

        Args:
            img_input: Input image

        Returns:
            Tuple of (pose results, annotated image)
        """
        results = self.model(
            img_input,
            conf=self.conf_thres,
            verbose=False
        )
        result = results[0]

        poses = []
        if result.keypoints is not None:
            for i, kpts in enumerate(result.keypoints):
                poses.append({
                    "person_id": i,
                    "keypoints": kpts.data.cpu().numpy().tolist(),
                    "confidence": float(result.boxes[i].conf[0]) if result.boxes else None
                })

        return poses, result.plot()

    def detect_segment(
        self,
        img_input: Union[str, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run instance segmentation (requires seg model).

        Args:
            img_input: Input image

        Returns:
            Tuple of (segmentation results, annotated image)
        """
        results = self.model(
            img_input,
            conf=self.conf_thres,
            verbose=False
        )
        result = results[0]

        segments = []
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                box = result.boxes[i]
                xyxy = box.xyxy[0].cpu().numpy()
                segments.append({
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x_min": int(xyxy[0]),
                        "y_min": int(xyxy[1]),
                        "x_max": int(xyxy[2]),
                        "y_max": int(xyxy[3])
                    },
                    "mask_shape": mask.data.shape
                })

        return segments, result.plot()
