"""
YOLOv11 (Ultralytics) object detection test script.
Uses shared modules from common/ and detectors/.
Supports: Object detection, Pose estimation, Segmentation, YOLO-World.
"""
import os
import sys

import ultralytics
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.config import Config
from common.utils import get_image_files, get_video_files, generate_output_path
from common.video_processor import VideoProcessor
from detectors.yolov11_detector import YOLOv11Detector


def batch_process_images(detector: YOLOv11Detector, input_dir: str, output_dir: str) -> None:
    """
    Process all images in directory.

    Args:
        detector: YOLOv11 detector instance
        input_dir: Input directory path
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    images = get_image_files(input_dir)
    print(f"Processing {len(images)} images...")

    for i, img_name in enumerate(images):
        img_path = f"{input_dir}/{img_name}"
        output_path = generate_output_path(img_path, output_dir)

        _, annotated = detector.detect(img_path)
        cv2.imwrite(output_path, annotated)

        print(f"[{i+1}/{len(images)}] {img_name} -> {output_path}")

        # Memory cleanup
        if (i + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nBatch processing complete. Results in: {output_dir}")


# Version info
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Environment check
ultralytics.checks()


# Configuration
OUTPUT_DIR = Config.get_output_dir("v11")
CONF_THRESHOLD = Config.CONF_THRESHOLD
IOU_THRESHOLD = Config.IOU_THRESHOLD
IMG_SIZE = Config.IMG_SIZE
MAX_DET = Config.MAX_DET

# Model selection
MODEL_NAME = "yolo11m"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

# Use custom classes from Config
ACTIVE_CLASSES = Config.CUSTOM_CLASSES_ROBOCUP


if __name__ == "__main__":
    # Load models
    model_yolo11 = YOLO(f"{MODEL_NAME}.pt")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Task: {model_yolo11.task}")
    print(f"Classes: {len(model_yolo11.names)} classes (COCO)")

    # YOLO-World model
    model_world = YOLO("yolov8m-world.pt")
    print(f"\nModel: YOLO-World")
    print(f"Task: {model_world.task}")

    # Create detector instances
    detector_yolo11 = YOLOv11Detector(
        model_yolo11,
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        imgsz=IMG_SIZE
    )

    detector_world = YOLOv11Detector(
        model_world,
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        custom_classes=ACTIVE_CLASSES
    )

    # Find test images
    images = get_image_files(Config.INPUT_DIR)
    if images:
        print(f"\nFound {len(images)} images in {Config.INPUT_DIR}:")
        for img in images:
            print(f"  - {img}")
        TEST_IMAGE = f"{Config.INPUT_DIR}/{images[0]}"
    else:
        TEST_IMAGE = None
        print(f"No images found in {Config.INPUT_DIR}")

    print(f"\nTest image: {TEST_IMAGE}")

    # YOLO11 detection test
    is_detect_yolo11 = False
    if is_detect_yolo11 and TEST_IMAGE:
        print("=" * 50)
        print("YOLO11 Detection (COCO 80 classes)")
        print("=" * 50)
        detections_yolo11 = detector_yolo11.detect_and_show(TEST_IMAGE)

    # YOLO-World detection test
    is_detect_yolo_world = False
    if is_detect_yolo_world and TEST_IMAGE:
        print("=" * 50)
        print(f"YOLO-World Detection (Custom {len(ACTIVE_CLASSES)} classes)")
        print("=" * 50)
        detections_world = detector_world.detect_and_show(TEST_IMAGE)

    # Batch image processing
    is_batch_image_test = True
    if is_batch_image_test:
        print("=" * 50)
        print("YOLO11 Batch Image Processing")
        print("=" * 50)
        batch_process_images(detector_yolo11, Config.INPUT_DIR, OUTPUT_DIR)

    # Dynamic class change demo
    is_yolo_world_dynamic_class_change = False
    if is_yolo_world_dynamic_class_change and TEST_IMAGE:
        print("=" * 50)
        print("YOLO-World: Dynamic Class Change Demo")
        print("=" * 50)

        # Search for specific objects
        specific_classes = ["cup", "tray"]
        detector_world.set_classes(specific_classes)
        print(f"\nSearching for: {specific_classes}")
        detector_world.detect_and_show(TEST_IMAGE)

        # Restore original classes
        detector_world.set_classes(ACTIVE_CLASSES)

    # Video processing test
    is_video_test = False
    if is_video_test:
        videos = get_video_files(Config.INPUT_DIR)
        if videos:
            print(f"\nFound {len(videos)} videos:")
            for vid in videos:
                print(f"  - {vid}")

            test_video = f"{Config.INPUT_DIR}/{videos[0]}"
            print(f"\nProcessing with YOLO11: {test_video}")

            processor = VideoProcessor(detector_yolo11, OUTPUT_DIR)
            processor.process_video(
                test_video,
                skip_frames=0,
                max_frames=None
            )
        else:
            print(f"No videos found in {Config.INPUT_DIR}")

    # Segmentation test
    is_yolo11_segmentation_test = False
    if is_yolo11_segmentation_test and TEST_IMAGE:
        print("=" * 50)
        print("YOLO11 Segmentation")
        print("=" * 50)

        detector_seg = YOLOv11Detector(
            "yolo11l-seg.pt",
            conf_thres=CONF_THRESHOLD
        )
        segments, annotated_seg = detector_seg.detect_segment(TEST_IMAGE)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_seg, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("YOLO11 Segmentation")
        plt.show()

        print(f"Found {len(segments)} segments")

    # Pose estimation test
    is_yolo11_pose_test = True
    if is_yolo11_pose_test:
        print("=" * 50)
        print("YOLO11 Pose Estimation")
        print("=" * 50)

        detector_pose = YOLOv11Detector(
            "yolo11m-pose.pt",
            conf_thres=CONF_THRESHOLD
        )

        pose_image = f"{Config.INPUT_DIR}/pose_sample.jpg"
        if os.path.exists(pose_image):
            poses, annotated_pose = detector_pose.detect_pose(pose_image)

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated_pose, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("YOLO11 Pose Estimation")
            plt.show()

            print(f"Found {len(poses)} persons")
        else:
            print(f"Pose sample image not found: {pose_image}")
