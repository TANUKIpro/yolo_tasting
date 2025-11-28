"""
Utility functions for YOLO detection systems.
"""
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt

from .config import Config


def get_image_files(directory: str) -> List[str]:
    """
    Get list of image files in directory.

    Args:
        directory: Directory path to search

    Returns:
        List of image file names
    """
    if not os.path.exists(directory):
        return []
    return [
        f for f in os.listdir(directory)
        if Path(f).suffix.lower() in Config.IMAGE_EXTENSIONS
    ]


def get_video_files(directory: str) -> List[str]:
    """
    Get list of video files in directory.

    Args:
        directory: Directory path to search

    Returns:
        List of video file names
    """
    if not os.path.exists(directory):
        return []
    return [
        f for f in os.listdir(directory)
        if Path(f).suffix.lower() in Config.VIDEO_EXTENSIONS
    ]


def print_detection_results(detections: List[Dict[str, Any]], title: str = "Detections") -> None:
    """
    Print detection results in formatted output.

    Args:
        detections: List of detection dictionaries
        title: Title for the output
    """
    print(f"\n{title} ({len(detections)}):")
    for i, d in enumerate(detections):
        class_name = d.get("class_name", "unknown")
        confidence = d.get("confidence", 0)
        x_min = d.get("x_min", 0)
        y_min = d.get("y_min", 0)
        x_max = d.get("x_max", 0)
        y_max = d.get("y_max", 0)
        print(f"  [{i}] {class_name}: {confidence:.2f} @ ({x_min}, {y_min}) - ({x_max}, {y_max})")


def show_detection_image(
    annotated_img,
    detections: List[Dict[str, Any]],
    figsize: tuple = (12, 8),
    title: str = None
) -> None:
    """
    Display detection results using matplotlib.

    Args:
        annotated_img: Annotated image (BGR format)
        detections: List of detection dictionaries
        figsize: Figure size for display
        title: Optional title (defaults to detection count)
    """
    if title is None:
        title = f"Detections: {len(detections)}"

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


def save_detection_image(annotated_img, output_path: str) -> str:
    """
    Save annotated image to file.

    Args:
        annotated_img: Annotated image (BGR format)
        output_path: Output file path

    Returns:
        Output path
    """
    cv2.imwrite(output_path, annotated_img)
    return output_path


def generate_output_path(input_path: str, output_dir: str, suffix: str = "_detected") -> str:
    """
    Generate output path based on input path.

    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename

    Returns:
        Generated output path
    """
    input_name = Path(input_path).stem
    ext = Path(input_path).suffix
    if ext.lower() in Config.VIDEO_EXTENSIONS:
        ext = ".mp4"
    elif ext.lower() in Config.IMAGE_EXTENSIONS:
        ext = ".jpg"
    return f"{output_dir}/{input_name}{suffix}{ext}"
