"""
Model registry for all supported YOLO models.

Defines model configurations, variants, and provides utilities
for model selection in the GUI.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class YOLOVersion(Enum):
    """Supported YOLO versions."""
    YOLOV5 = "yolov5"
    YOLOV11 = "yolov11"
    YOLO_WORLD = "yolo_world"


class TaskType(Enum):
    """Supported task types."""
    DETECTION = "detect"
    POSE = "pose"
    SEGMENTATION = "seg"
    CLASSIFICATION = "cls"
    OBB = "obb"  # Oriented Bounding Box


class ModelSize(Enum):
    """Model size variants."""
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    version: YOLOVersion
    task: TaskType
    size: ModelSize
    model_file: str
    display_name: str
    supports_custom_classes: bool = False
    description: str = ""


@dataclass
class ModelRegistry:
    """Registry of all available YOLO models."""

    # YOLOv5 models (detection only, uses PyTorch Hub or local weights)
    YOLOV5_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolov5n": ModelConfig(
            YOLOVersion.YOLOV5, TaskType.DETECTION, ModelSize.NANO,
            "yolov5n.pt", "YOLOv5 Nano",
            description="Fastest, smallest model (~1.9M params)"
        ),
        "yolov5s": ModelConfig(
            YOLOVersion.YOLOV5, TaskType.DETECTION, ModelSize.SMALL,
            "yolov5s.pt", "YOLOv5 Small",
            description="Balanced speed/accuracy (~7.2M params)"
        ),
        "yolov5m": ModelConfig(
            YOLOVersion.YOLOV5, TaskType.DETECTION, ModelSize.MEDIUM,
            "yolov5m.pt", "YOLOv5 Medium",
            description="Higher accuracy (~21.2M params)"
        ),
        "yolov5l": ModelConfig(
            YOLOVersion.YOLOV5, TaskType.DETECTION, ModelSize.LARGE,
            "yolov5l.pt", "YOLOv5 Large",
            description="High accuracy (~46.5M params)"
        ),
        "yolov5x": ModelConfig(
            YOLOVersion.YOLOV5, TaskType.DETECTION, ModelSize.XLARGE,
            "yolov5x.pt", "YOLOv5 XLarge",
            description="Highest accuracy (~86.7M params)"
        ),
    })

    # YOLOv11 Detection models
    YOLOV11_DETECT_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolo11n": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.DETECTION, ModelSize.NANO,
            "yolo11n.pt", "YOLO11 Nano",
            description="Fastest YOLO11 (~2.6M params)"
        ),
        "yolo11s": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.DETECTION, ModelSize.SMALL,
            "yolo11s.pt", "YOLO11 Small",
            description="Balanced (~9.4M params)"
        ),
        "yolo11m": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.DETECTION, ModelSize.MEDIUM,
            "yolo11m.pt", "YOLO11 Medium",
            description="Better accuracy (~20.1M params)"
        ),
        "yolo11l": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.DETECTION, ModelSize.LARGE,
            "yolo11l.pt", "YOLO11 Large",
            description="High accuracy (~25.3M params)"
        ),
        "yolo11x": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.DETECTION, ModelSize.XLARGE,
            "yolo11x.pt", "YOLO11 XLarge",
            description="Highest accuracy (~56.9M params)"
        ),
    })

    # YOLOv11 Pose models
    YOLOV11_POSE_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolo11n-pose": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.POSE, ModelSize.NANO,
            "yolo11n-pose.pt", "YOLO11 Pose Nano",
            description="Fast pose estimation"
        ),
        "yolo11s-pose": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.POSE, ModelSize.SMALL,
            "yolo11s-pose.pt", "YOLO11 Pose Small",
            description="Balanced pose estimation"
        ),
        "yolo11m-pose": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.POSE, ModelSize.MEDIUM,
            "yolo11m-pose.pt", "YOLO11 Pose Medium",
            description="Accurate pose estimation"
        ),
        "yolo11l-pose": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.POSE, ModelSize.LARGE,
            "yolo11l-pose.pt", "YOLO11 Pose Large",
            description="High accuracy pose"
        ),
        "yolo11x-pose": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.POSE, ModelSize.XLARGE,
            "yolo11x-pose.pt", "YOLO11 Pose XLarge",
            description="Highest accuracy pose"
        ),
    })

    # YOLOv11 Segmentation models
    YOLOV11_SEG_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolo11n-seg": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.SEGMENTATION, ModelSize.NANO,
            "yolo11n-seg.pt", "YOLO11 Seg Nano",
            description="Fast instance segmentation"
        ),
        "yolo11s-seg": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.SEGMENTATION, ModelSize.SMALL,
            "yolo11s-seg.pt", "YOLO11 Seg Small",
            description="Balanced segmentation"
        ),
        "yolo11m-seg": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.SEGMENTATION, ModelSize.MEDIUM,
            "yolo11m-seg.pt", "YOLO11 Seg Medium",
            description="Accurate segmentation"
        ),
        "yolo11l-seg": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.SEGMENTATION, ModelSize.LARGE,
            "yolo11l-seg.pt", "YOLO11 Seg Large",
            description="High accuracy segmentation"
        ),
        "yolo11x-seg": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.SEGMENTATION, ModelSize.XLARGE,
            "yolo11x-seg.pt", "YOLO11 Seg XLarge",
            description="Highest accuracy segmentation"
        ),
    })

    # YOLOv11 Classification models
    YOLOV11_CLS_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolo11n-cls": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.CLASSIFICATION, ModelSize.NANO,
            "yolo11n-cls.pt", "YOLO11 Cls Nano",
            description="Fast classification"
        ),
        "yolo11s-cls": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.CLASSIFICATION, ModelSize.SMALL,
            "yolo11s-cls.pt", "YOLO11 Cls Small",
            description="Balanced classification"
        ),
        "yolo11m-cls": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.CLASSIFICATION, ModelSize.MEDIUM,
            "yolo11m-cls.pt", "YOLO11 Cls Medium",
            description="Accurate classification"
        ),
        "yolo11l-cls": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.CLASSIFICATION, ModelSize.LARGE,
            "yolo11l-cls.pt", "YOLO11 Cls Large",
            description="High accuracy classification"
        ),
        "yolo11x-cls": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.CLASSIFICATION, ModelSize.XLARGE,
            "yolo11x-cls.pt", "YOLO11 Cls XLarge",
            description="Highest accuracy classification"
        ),
    })

    # YOLOv11 OBB (Oriented Bounding Box) models
    YOLOV11_OBB_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolo11n-obb": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.OBB, ModelSize.NANO,
            "yolo11n-obb.pt", "YOLO11 OBB Nano",
            description="Fast oriented detection"
        ),
        "yolo11s-obb": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.OBB, ModelSize.SMALL,
            "yolo11s-obb.pt", "YOLO11 OBB Small",
            description="Balanced oriented detection"
        ),
        "yolo11m-obb": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.OBB, ModelSize.MEDIUM,
            "yolo11m-obb.pt", "YOLO11 OBB Medium",
            description="Accurate oriented detection"
        ),
        "yolo11l-obb": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.OBB, ModelSize.LARGE,
            "yolo11l-obb.pt", "YOLO11 OBB Large",
            description="High accuracy OBB"
        ),
        "yolo11x-obb": ModelConfig(
            YOLOVersion.YOLOV11, TaskType.OBB, ModelSize.XLARGE,
            "yolo11x-obb.pt", "YOLO11 OBB XLarge",
            description="Highest accuracy OBB"
        ),
    })

    # YOLO-World models (open-vocabulary detection with custom classes)
    YOLO_WORLD_MODELS: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "yolov8s-world": ModelConfig(
            YOLOVersion.YOLO_WORLD, TaskType.DETECTION, ModelSize.SMALL,
            "yolov8s-world.pt", "YOLO-World Small",
            supports_custom_classes=True,
            description="Fast open-vocabulary detection"
        ),
        "yolov8m-world": ModelConfig(
            YOLOVersion.YOLO_WORLD, TaskType.DETECTION, ModelSize.MEDIUM,
            "yolov8m-world.pt", "YOLO-World Medium",
            supports_custom_classes=True,
            description="Balanced open-vocabulary detection"
        ),
        "yolov8l-world": ModelConfig(
            YOLOVersion.YOLO_WORLD, TaskType.DETECTION, ModelSize.LARGE,
            "yolov8l-world.pt", "YOLO-World Large",
            supports_custom_classes=True,
            description="Accurate open-vocabulary detection"
        ),
        "yolov8x-world": ModelConfig(
            YOLOVersion.YOLO_WORLD, TaskType.DETECTION, ModelSize.XLARGE,
            "yolov8x-world.pt", "YOLO-World XLarge",
            supports_custom_classes=True,
            description="Highest accuracy open-vocabulary"
        ),
    })

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Get all registered models."""
        all_models = {}
        all_models.update(self.YOLOV5_MODELS)
        all_models.update(self.YOLOV11_DETECT_MODELS)
        all_models.update(self.YOLOV11_POSE_MODELS)
        all_models.update(self.YOLOV11_SEG_MODELS)
        all_models.update(self.YOLOV11_CLS_MODELS)
        all_models.update(self.YOLOV11_OBB_MODELS)
        all_models.update(self.YOLO_WORLD_MODELS)
        return all_models


# Global registry instance
REGISTRY = ModelRegistry()


def get_versions() -> List[str]:
    """Get list of YOLO versions for GUI dropdown."""
    return ["YOLOv5", "YOLOv11", "YOLO-World"]


def get_tasks_for_version(version: str) -> List[str]:
    """Get available tasks for a specific YOLO version."""
    if version == "YOLOv5":
        return ["Detection"]
    elif version == "YOLOv11":
        return ["Detection", "Pose", "Segmentation", "Classification", "OBB"]
    elif version == "YOLO-World":
        return ["Detection (Custom Classes)"]
    return []


def get_sizes_for_version_task(version: str, task: str) -> List[str]:
    """Get available model sizes for a version/task combination."""
    if version == "YOLOv5":
        return ["Nano", "Small", "Medium", "Large", "XLarge"]
    elif version == "YOLO-World":
        return ["Small", "Medium", "Large", "XLarge"]
    elif version == "YOLOv11":
        return ["Nano", "Small", "Medium", "Large", "XLarge"]
    return []


def get_model_key(version: str, task: str, size: str) -> Optional[str]:
    """
    Get the model key from version, task, and size selections.

    Args:
        version: "YOLOv5", "YOLOv11", or "YOLO-World"
        task: Task type string
        size: Size string (Nano, Small, etc.)

    Returns:
        Model key string (e.g., "yolo11m-pose") or None
    """
    size_map = {
        "Nano": "n", "Small": "s", "Medium": "m",
        "Large": "l", "XLarge": "x"
    }
    size_code = size_map.get(size, "m")

    if version == "YOLOv5":
        return f"yolov5{size_code}"

    elif version == "YOLO-World":
        return f"yolov8{size_code}-world"

    elif version == "YOLOv11":
        task_suffix_map = {
            "Detection": "",
            "Pose": "-pose",
            "Segmentation": "-seg",
            "Classification": "-cls",
            "OBB": "-obb"
        }
        suffix = task_suffix_map.get(task, "")
        return f"yolo11{size_code}{suffix}"

    return None


def get_model_config(model_key: str) -> Optional[ModelConfig]:
    """Get model configuration by key."""
    all_models = REGISTRY.get_all_models()
    return all_models.get(model_key)


# Preset class lists for YOLO-World
CLASS_PRESETS = {
    "COCO (80 classes)": None,  # Use default COCO classes
    "RoboCup@Home": [
        "person", "cup", "bottle", "bowl", "plate",
        "fork", "knife", "spoon", "banana", "apple",
        "orange", "sandwich", "book", "cell phone", "remote",
        "laptop", "chair", "couch", "dining table", "potted plant"
    ],
    "YCB Objects": [
        "cracker box", "sugar box", "tomato soup can", "mustard bottle",
        "tuna fish can", "pudding box", "gelatin box", "potted meat can",
        "banana", "strawberry", "apple", "lemon", "peach", "pear", "orange", "plum",
        "pitcher", "bowl", "mug", "plate", "fork", "spoon", "knife", "spatula",
        "sponge", "power drill", "wood block", "scissors", "marker", "clamp",
        "tennis ball", "golf ball", "baseball", "dice", "rubiks cube"
    ],
    "Simple (6 classes)": [
        "person", "cup", "bottle", "book", "phone", "laptop"
    ],
    "Custom": []  # User-defined classes
}


def get_class_presets() -> List[str]:
    """Get list of class preset names for dropdown."""
    return list(CLASS_PRESETS.keys())


def get_preset_classes(preset_name: str) -> Optional[List[str]]:
    """Get class list for a preset."""
    return CLASS_PRESETS.get(preset_name)
