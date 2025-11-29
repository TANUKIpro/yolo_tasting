"""
Detector manager for dynamic model loading and caching.

Handles loading, caching, and parameter updates for YOLO detectors.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .model_registry import (
    YOLOVersion, TaskType, ModelConfig,
    get_model_key, get_model_config, get_preset_classes
)


@dataclass
class DetectionParams:
    """Parameters for detection."""
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    img_size: int = 640
    max_det: int = 300


class DetectorManager:
    """
    Manages YOLO detector instances with caching and parameter updates.

    Supports YOLOv5, YOLOv11, and YOLO-World models with dynamic loading.
    """

    def __init__(self):
        """Initialize the detector manager."""
        self._detector = None
        self._current_model_key: Optional[str] = None
        self._current_config: Optional[ModelConfig] = None
        self._params = DetectionParams()
        self._custom_classes: Optional[List[str]] = None
        self._yolov5_available = self._check_yolov5_available()

    def _check_yolov5_available(self) -> bool:
        """Check if YOLOv5 is available."""
        try:
            from yolov5.models.experimental import attempt_load
            return True
        except ImportError:
            return False

    @property
    def params(self) -> DetectionParams:
        """Get current detection parameters."""
        return self._params

    @property
    def current_model_key(self) -> Optional[str]:
        """Get current model key."""
        return self._current_model_key

    @property
    def is_loaded(self) -> bool:
        """Check if a detector is loaded."""
        return self._detector is not None

    def update_params(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        img_size: Optional[int] = None,
        max_det: Optional[int] = None
    ) -> None:
        """
        Update detection parameters.

        Args:
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: NMS IoU threshold (0-1)
            img_size: Inference image size
            max_det: Maximum detections
        """
        if conf_threshold is not None:
            self._params.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self._params.iou_threshold = iou_threshold
        if img_size is not None:
            self._params.img_size = img_size
        if max_det is not None:
            self._params.max_det = max_det

        # Update detector parameters if loaded
        if self._detector is not None:
            self._detector.conf_thres = self._params.conf_threshold
            self._detector.iou_thres = self._params.iou_threshold
            self._detector.imgsz = self._params.img_size
            self._detector.max_det = self._params.max_det

    def set_custom_classes(self, classes: Optional[List[str]]) -> None:
        """
        Set custom classes for YOLO-World.

        Args:
            classes: List of class names, or None for default COCO
        """
        self._custom_classes = classes

        # Update detector if it's YOLO-World
        if (self._detector is not None and
            self._current_config is not None and
            self._current_config.version == YOLOVersion.YOLO_WORLD and
            classes is not None):
            self._detector.set_classes(classes)

    def load_model(
        self,
        version: str,
        task: str,
        size: str,
        custom_classes: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Load a YOLO model based on GUI selections.

        Args:
            version: "YOLOv5", "YOLOv11", or "YOLO-World"
            task: Task type string
            size: Size string (Nano, Small, etc.)
            custom_classes: Custom classes for YOLO-World

        Returns:
            Tuple of (success, message)
        """
        # Get model key from selections
        model_key = get_model_key(version, task, size)
        if model_key is None:
            return False, f"Invalid model selection: {version}/{task}/{size}"

        # Check if already loaded
        if model_key == self._current_model_key and self._detector is not None:
            # Just update custom classes if needed
            if custom_classes is not None and self._current_config.supports_custom_classes:
                self.set_custom_classes(custom_classes)
            return True, f"Model already loaded: {model_key}"

        # Get model config
        config = get_model_config(model_key)
        if config is None:
            return False, f"Model configuration not found: {model_key}"

        # Clear previous detector
        self._clear_detector()

        # Load appropriate detector
        try:
            if config.version == YOLOVersion.YOLOV5:
                success, msg = self._load_yolov5(config)
            elif config.version in (YOLOVersion.YOLOV11, YOLOVersion.YOLO_WORLD):
                success, msg = self._load_yolov11(config, custom_classes)
            else:
                return False, f"Unsupported YOLO version: {config.version}"

            if success:
                self._current_model_key = model_key
                self._current_config = config

            return success, msg

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def _load_yolov5(self, config: ModelConfig) -> Tuple[bool, str]:
        """Load YOLOv5 detector using Ultralytics."""
        try:
            from detectors.yolov11_detector import YOLOv11Detector

            # YOLOv5 is also supported by Ultralytics YOLO class
            # Use yolov5nu (Ultralytics format) instead of legacy format
            model_map = {
                "yolov5n.pt": "yolov5nu.pt",
                "yolov5s.pt": "yolov5su.pt",
                "yolov5m.pt": "yolov5mu.pt",
                "yolov5l.pt": "yolov5lu.pt",
                "yolov5x.pt": "yolov5xu.pt",
            }
            model_file = model_map.get(config.model_file, config.model_file)

            self._detector = YOLOv11Detector(
                model=model_file,
                conf_thres=self._params.conf_threshold,
                iou_thres=self._params.iou_threshold,
                imgsz=self._params.img_size,
                max_det=self._params.max_det,
                custom_classes=None
            )
            return True, f"Loaded {config.display_name}"

        except Exception as e:
            return False, f"YOLOv5 load error: {str(e)}"

    def _load_yolov11(
        self,
        config: ModelConfig,
        custom_classes: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Load YOLOv11 or YOLO-World detector."""
        try:
            from detectors.yolov11_detector import YOLOv11Detector

            # Determine if we need custom classes (YOLO-World)
            classes = custom_classes if config.supports_custom_classes else None

            self._detector = YOLOv11Detector(
                model=config.model_file,
                conf_thres=self._params.conf_threshold,
                iou_thres=self._params.iou_threshold,
                imgsz=self._params.img_size,
                max_det=self._params.max_det,
                custom_classes=classes
            )

            self._custom_classes = classes
            return True, f"Loaded {config.display_name}"

        except Exception as e:
            return False, f"YOLO11 load error: {str(e)}"

    def _clear_detector(self) -> None:
        """Clear current detector and free memory."""
        if self._detector is not None:
            del self._detector
            self._detector = None

        self._current_model_key = None
        self._current_config = None

        # Clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(
        self,
        image: Union[str, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], str]:
        """
        Run detection on an image.

        Args:
            image: Image path or numpy array (BGR)

        Returns:
            Tuple of (detections list, annotated image, status message)
        """
        if self._detector is None:
            return [], None, "No model loaded"

        if self._current_config is None:
            return [], None, "Model configuration not available"

        try:
            task = self._current_config.task

            # Run appropriate detection method based on task
            if task == TaskType.DETECTION:
                detections, annotated = self._detector.detect(image)
                return detections, annotated, f"Detected {len(detections)} objects"

            elif task == TaskType.POSE:
                poses, annotated = self._detector.detect_pose(image)
                return poses, annotated, f"Detected {len(poses)} poses"

            elif task == TaskType.SEGMENTATION:
                segments, annotated = self._detector.detect_segment(image)
                return segments, annotated, f"Detected {len(segments)} segments"

            elif task == TaskType.CLASSIFICATION:
                # Classification returns different format
                results, annotated = self._run_classification(image)
                return results, annotated, f"Classification complete"

            elif task == TaskType.OBB:
                # OBB uses standard detect but with oriented boxes
                detections, annotated = self._detector.detect(image)
                return detections, annotated, f"Detected {len(detections)} oriented objects"

            else:
                return [], None, f"Unsupported task: {task}"

        except Exception as e:
            return [], None, f"Detection error: {str(e)}"

    def _run_classification(
        self,
        image: Union[str, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Run classification (special handling)."""
        from ultralytics import YOLO
        import cv2

        # Run inference
        results = self._detector.model(
            image,
            verbose=False
        )
        result = results[0]

        # Get top-5 predictions
        probs = result.probs
        classifications = []

        if probs is not None:
            top5_indices = probs.top5
            top5_confs = probs.top5conf

            for idx, conf in zip(top5_indices, top5_confs):
                classifications.append({
                    "class_id": int(idx),
                    "class_name": result.names[int(idx)],
                    "confidence": float(conf)
                })

        # Get annotated image
        annotated = result.plot()

        return classifications, annotated

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self._current_config is None:
            return {"status": "No model loaded"}

        return {
            "model_key": self._current_model_key,
            "display_name": self._current_config.display_name,
            "version": self._current_config.version.value,
            "task": self._current_config.task.value,
            "size": self._current_config.size.value,
            "supports_custom_classes": self._current_config.supports_custom_classes,
            "description": self._current_config.description,
            "params": {
                "conf_threshold": self._params.conf_threshold,
                "iou_threshold": self._params.iou_threshold,
                "img_size": self._params.img_size,
                "max_det": self._params.max_det
            }
        }


# Global detector manager instance
_manager: Optional[DetectorManager] = None


def get_detector_manager() -> DetectorManager:
    """Get or create the global detector manager instance."""
    global _manager
    if _manager is None:
        _manager = DetectorManager()
    return _manager
