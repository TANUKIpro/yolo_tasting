"""
Abstract base class for YOLO detectors.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt

from common.utils import print_detection_results, show_detection_image


class BaseDetector(ABC):
    """
    Abstract base class for YOLO object detectors.
    All detector implementations should inherit from this class.
    """

    def __init__(
        self,
        conf_thres: float = 0.5,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300
    ):
        """
        Initialize base detector.

        Args:
            conf_thres: Confidence threshold
            iou_thres: NMS IoU threshold
            imgsz: Inference image size
            max_det: Maximum detections
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.max_det = max_det

    @abstractmethod
    def detect(
        self,
        img_input: Union[str, np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run object detection on input image.

        Args:
            img_input: Input image (numpy array BGR or file path)

        Returns:
            Tuple of (detections list, annotated image)
            Each detection dict contains:
                - x_min, y_min, x_max, y_max: Bounding box coordinates
                - confidence: Detection confidence
                - class_name: Detected class name
        """
        pass

    @property
    @abstractmethod
    def names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        pass

    def detect_and_show(
        self,
        source: Union[str, np.ndarray],
        figsize: tuple = (12, 8)
    ) -> List[Dict[str, Any]]:
        """
        Detect objects and display results.

        Args:
            source: Input image path or numpy array
            figsize: Figure size for display

        Returns:
            List of detection dictionaries
        """
        detections, annotated_img = self.detect(source)
        show_detection_image(annotated_img, detections, figsize=figsize)
        print_detection_results(detections)
        return detections

    def detect_and_save(
        self,
        source: Union[str, np.ndarray],
        output_path: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Detect objects and save result image.

        Args:
            source: Input image path or numpy array
            output_path: Output file path

        Returns:
            Tuple of (detections list, output path)
        """
        detections, annotated_img = self.detect(source)
        cv2.imwrite(output_path, annotated_img)
        print(f"Result saved to: {output_path}")
        return detections, output_path
