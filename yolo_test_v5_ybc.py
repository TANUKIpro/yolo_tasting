"""
YOLOv5 object detection test script.
Uses shared modules from common/ and detectors/.
"""
import os
import sys

import torch
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.config import Config
from common.utils import (
    get_image_files,
    get_video_files,
    print_detection_results,
    generate_output_path,
)
from common.video_processor import VideoProcessor
from detectors.yolov5_detector import YOLOv5Detector


# Version info
import yolov5
print(f"YOLOv5 version: {yolov5.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# Configuration
OUTPUT_DIR = Config.get_output_dir("v5")
WEIGHTS_DIR = Config.get_weights_dir("v5")
WEIGHTS_PATH = f"{WEIGHTS_DIR}/ycb.pt"

# Detection parameters
CONF_THRESHOLD = Config.CONF_THRESHOLD
IOU_THRESHOLD = Config.IOU_THRESHOLD
IMG_SIZE = [640, 480]
MAX_DET = 1000


def batch_process_images(detector: YOLOv5Detector, input_dir: str, output_dir: str) -> None:
    """
    Process all images in directory.

    Args:
        detector: YOLOv5 detector instance
        input_dir: Input directory path
        output_dir: Output directory path
    """
    images = get_image_files(input_dir)
    print(f"Processing {len(images)} images...")

    for i, img_name in enumerate(images):
        img_path = f"{input_dir}/{img_name}"
        output_path = generate_output_path(img_path, output_dir)

        _, annotated = detector.detect(img_path)
        cv2.imwrite(output_path, annotated)

        print(f"[{i+1}/{len(images)}] {img_name} -> {output_path}")

        # Memory cleanup
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    print(f"\nBatch processing complete. Results in: {output_dir}")


if __name__ == "__main__":
    # Initialize detector
    if os.path.exists(WEIGHTS_PATH):
        detector = YOLOv5Detector(
            weights=WEIGHTS_PATH,
            imgsz=IMG_SIZE,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            max_det=MAX_DET
        )
    else:
        print(f"ERROR: Weights file not found: {WEIGHTS_PATH}")
        print(f"Please upload your weights to: {WEIGHTS_DIR}")
        print("\nOr download pretrained weights:")
        print(f"  wget -P {WEIGHTS_DIR} https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt")
        sys.exit(1)

    # Single image test
    is_image_test = False
    if is_image_test:
        images = get_image_files(Config.INPUT_DIR)
        if images:
            print(f"Found {len(images)} images in {Config.INPUT_DIR}:")
            for img in images:
                print(f"  - {img}")

            test_image = f"{Config.INPUT_DIR}/{images[0]}"
            print(f"\nTesting with: {test_image}")
            detector.detect_and_show(test_image)

    # Batch image processing
    is_batch_image_test = True
    if is_batch_image_test:
        batch_process_images(detector, Config.INPUT_DIR, OUTPUT_DIR)

    # Video processing test
    is_video_test = False
    if is_video_test:
        videos = get_video_files(Config.INPUT_DIR)
        if videos:
            print(f"Found {len(videos)} videos in {Config.INPUT_DIR}:")
            for vid in videos:
                print(f"  - {vid}")

            test_video = f"{Config.INPUT_DIR}/{videos[0]}"
            print(f"\nProcessing: {test_video}")

            processor = VideoProcessor(detector, OUTPUT_DIR)
            processor.process_video(
                test_video,
                skip_frames=0,
                max_frames=None
            )
        else:
            print(f"No videos found in {Config.INPUT_DIR}")
