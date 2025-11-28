"""
Shared configuration for YOLO detection systems.
"""
from pathlib import Path


class Config:
    """Configuration class for YOLO detection parameters."""

    # Base directories
    DRIVE_BASE = "/home/ryo/workspace/yolo_tasting"
    INPUT_DIR = f"{DRIVE_BASE}/common/input"

    # Detection parameters (defaults)
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    IMG_SIZE = 640
    MAX_DET = 300

    # File extensions
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    # YOLO-World preset classes
    CUSTOM_CLASSES_ROBOCUP = [
        "person", "cup", "bottle", "bowl", "plate",
        "fork", "knife", "spoon", "banana", "apple",
        "orange", "sandwich", "book", "cell phone", "remote",
        "laptop", "chair", "couch", "dining table", "potted plant"
    ]

    CUSTOM_CLASSES_YCB = [
        "cracker box", "sugar box", "tomato soup can", "mustard bottle",
        "tuna fish can", "pudding box", "gelatin box", "potted meat can",
        "banana", "strawberry", "apple", "lemon", "peach", "pear", "orange", "plum",
        "pitcher", "bowl", "mug", "plate", "fork", "spoon", "knife", "spatula",
        "sponge", "power drill", "wood block", "scissors", "marker", "clamp",
        "tennis ball", "golf ball", "baseball", "dice", "rubiks cube"
    ]

    CUSTOM_CLASSES_SIMPLE = [
        "person", "cup", "bottle", "book", "phone", "laptop"
    ]

    @classmethod
    def get_output_dir(cls, version: str) -> str:
        """Get output directory for specific YOLO version."""
        return f"{cls.DRIVE_BASE}/yolo_{version}/output"

    @classmethod
    def get_weights_dir(cls, version: str) -> str:
        """Get weights directory for specific YOLO version."""
        return f"{cls.DRIVE_BASE}/yolo_{version}/weights"
