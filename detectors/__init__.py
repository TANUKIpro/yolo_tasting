"""
YOLO detector implementations.
"""
from .base import BaseDetector
from .yolov5_detector import YOLOv5Detector
from .yolov11_detector import YOLOv11Detector

__all__ = [
    "BaseDetector",
    "YOLOv5Detector",
    "YOLOv11Detector",
]
