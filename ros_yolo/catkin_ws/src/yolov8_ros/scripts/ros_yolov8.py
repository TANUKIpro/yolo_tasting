#!/usr/bin/env python3
"""
YOLOv8 ROS Node - Ultralytics API
Subscribes to camera images and publishes object detections.
"""
import argparse
import numpy as np
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rospy

import torch
from ultralytics import YOLO

from sensor_msgs.msg import Image
from yolov8_ros.msg import RecognitionObject, RecognitionObjectArray

bridge = CvBridge()


class Detector:
    """YOLOv8 Object Detector for ROS."""

    def __init__(
        self,
        weights='yolov8n.pt',
        imgsz=640,
        conf_thres=0.5,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        classes=None,
        agnostic_nms=False,  # Kept for argument compatibility
        augment=False,       # Kept for argument compatibility
        visualize=False,     # Kept for argument compatibility
        update=False,        # Kept for argument compatibility
        line_thickness=3,
        hide_labels=False,   # Kept for argument compatibility
        hide_conf=False,     # Kept for argument compatibility
        half=False,          # Kept for argument compatibility
        topic='image'
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            weights: Path to YOLOv8 model weights (.pt file)
            imgsz: Inference image size (pixels)
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            max_det: Maximum detections per image
            device: Device to run inference on ('', '0', 'cpu')
            view_img: Display annotated images in window
            classes: Filter by class indices (e.g., [0, 2, 3])
            topic: ROS topic to subscribe to
        """
        # Device selection
        if device == '':
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Detection parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.max_det = max_det
        self.imgsz = imgsz
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.sub_topic_name = topic

        # Load YOLOv8 model
        w = weights[0] if isinstance(weights, list) else weights
        rospy.loginfo(f"Loading YOLOv8 model: {w}")

        try:
            self.model = YOLO(w)
            rospy.loginfo(f"YOLOv8 model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load model {w}: {e}")
            rospy.logwarn("Falling back to yolov8n.pt")
            self.model = YOLO('yolov8n.pt')

        # Get class names
        self.names = self.model.names
        rospy.loginfo(f"Device: {self.device}")
        rospy.loginfo(f"Classes: {len(self.names)} ({list(self.names.values())[:5]}...)")

        # ROS publishers and subscribers
        self.sub = rospy.Subscriber(self.sub_topic_name, Image, self.image_callback)
        self.pub = rospy.Publisher("yolov8_obj", RecognitionObjectArray, queue_size=1)
        rospy.loginfo(f"Subscribed to: {self.sub_topic_name}")
        rospy.loginfo(f"Publishing to: /yolov8_obj")

    def process_img(self, cv_image):
        """
        Process image with YOLOv8.

        Args:
            cv_image: OpenCV BGR image (numpy array)

        Returns:
            tuple: (result_string, RecognitionObjectArray)
        """
        # Run YOLOv8 inference
        # YOLOv8 automatically handles all preprocessing:
        # - Letterbox resize
        # - BGR to RGB conversion
        # - HWC to CHW transpose
        # - Tensor conversion and normalization
        # - NMS filtering
        results = self.model(
            cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False
        )

        result = results[0]  # Get first result (single image)

        # Parse detections
        recog_obj_arr = RecognitionObjectArray()
        result_str = ""

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Extract box data - coordinates already scaled to original image size
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.names[cls_id]

                # Create ROS message
                recog_obj = RecognitionObject()
                recog_obj.x_min = int(xyxy[0])
                recog_obj.y_min = int(xyxy[1])
                recog_obj.x_max = int(xyxy[2])
                recog_obj.y_max = int(xyxy[3])
                recog_obj.confidence = conf
                recog_obj.class_name = cls_name
                recog_obj_arr.array.append(recog_obj)

                # Build result string
                result_str += f'{int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])},{cls_name},{conf:.2f} '

        # Optional visualization
        if self.view_img:
            annotated_img = result.plot(line_width=self.line_thickness)
            cv2.imshow("YOLOv8 Detection", annotated_img)
            cv2.waitKey(1)

        return result_str, recog_obj_arr

    def image_callback(self, msg):
        """
        ROS callback for image messages.

        Args:
            msg: sensor_msgs/Image message
        """
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        try:
            pub_msg, tmp_msg = self.process_img(cv_image)
            tmp_msg.header = msg.header
            self.pub.publish(tmp_msg)
        except Exception as e:
            rospy.logerr(f"Detection error: {e}")


if __name__ == "__main__":
    # Argument parser (maintains compatibility with YOLOv5 arguments)
    parser = argparse.ArgumentParser(description='YOLOv8 ROS Node')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov8n.pt',
                        help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view_img', action='store_true',
                        help='show results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS (kept for compatibility)')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference (kept for compatibility)')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features (kept for compatibility)')
    parser.add_argument('--update', action='store_true',
                        help='update all models (kept for compatibility)')
    parser.add_argument('--line-thickness', default=3, type=int,
                        help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true',
                        help='hide labels (kept for compatibility)')
    parser.add_argument('--hide-conf', default=False, action='store_true',
                        help='hide confidences (kept for compatibility)')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference (kept for compatibility)')
    parser.add_argument('--topic', type=str, default='/camera/rgb/image_rect_color',
                        help='ros topic to subscribe to')

    opt = parser.parse_args()

    # Initialize ROS node first
    rospy.init_node("yolov8_detector", anonymous=False)
    rospy.loginfo("=== YOLOv8 ROS Node Starting ===")

    # Create detector
    detector = Detector(**vars(opt))

    # Spin
    rospy.loginfo("YOLOv8 detector ready, spinning...")
    rospy.spin()
