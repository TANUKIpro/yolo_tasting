"""
GUI module for YOLO Tasting.

Provides a Gradio-based graphical interface for model selection,
parameter tuning, and real-time object detection visualization.
"""

from .app import create_app, launch_app

__all__ = ["create_app", "launch_app"]
