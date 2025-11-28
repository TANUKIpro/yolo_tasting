"""
YOLO detector implementations.

Use lazy imports to avoid dependency issues:
    from detectors.base import BaseDetector
    from detectors.yolov11_detector import YOLOv11Detector
    from detectors.yolov5_detector import YOLOv5Detector  # requires yolov5 package
"""
from .base import BaseDetector

__all__ = [
    "BaseDetector",
    "YOLOv5Detector",
    "YOLOv11Detector",
]


def __getattr__(name):
    """Lazy import for detector classes."""
    if name == "YOLOv5Detector":
        from .yolov5_detector import YOLOv5Detector
        return YOLOv5Detector
    elif name == "YOLOv11Detector":
        from .yolov11_detector import YOLOv11Detector
        return YOLOv11Detector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
