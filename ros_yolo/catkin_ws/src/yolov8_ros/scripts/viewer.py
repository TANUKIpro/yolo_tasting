#!/usr/bin/env python3
"""
YOLOv5検出結果ビューアー

カメラ画像と検出結果を同時購読し、
検出ボックスをオーバーレイしてOpenCVで表示する。
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from yolov5_ros.msg import RecognitionObjectArray
import threading


class YOLOViewer:
    def __init__(self):
        rospy.init_node('yolo_viewer', anonymous=True)

        # パラメータ
        self.image_topic = rospy.get_param(
            '~image_topic',
            '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        )
        self.detection_topic = rospy.get_param(
            '~detection_topic',
            '/yolov5_obj'
        )

        # CV Bridge
        self.bridge = CvBridge()

        # 最新データ保持
        self.current_image = None
        self.current_detections = None
        self.lock = threading.Lock()

        # カラーパレット（クラスごとに異なる色）
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

        # サブスクライバ
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1
        )
        self.detection_sub = rospy.Subscriber(
            self.detection_topic, RecognitionObjectArray, self.detection_callback, queue_size=1
        )

        rospy.loginfo(f"YOLOViewer started")
        rospy.loginfo(f"  Image topic: {self.image_topic}")
        rospy.loginfo(f"  Detection topic: {self.detection_topic}")

    def image_callback(self, msg):
        """カメラ画像を受信"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.current_image = cv_image.copy()
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def detection_callback(self, msg):
        """検出結果を受信"""
        with self.lock:
            self.current_detections = msg

    def draw_detections(self, image, detections):
        """画像に検出ボックスを描画"""
        if detections is None:
            return image

        for det in detections.array:
            # 色を取得（クラス名のハッシュ値で決定）
            color_idx = hash(det.class_name) % 100
            color = tuple(int(c) for c in self.colors[color_idx])

            # バウンディングボックス描画
            x1, y1, x2, y2 = det.x_min, det.y_min, det.x_max, det.y_max
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # ラベル描画
            label = f"{det.class_name}: {det.confidence:.2f}"
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # ラベル背景
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )

            # ラベルテキスト
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

        return image

    def run(self):
        """メインループ"""
        rate = rospy.Rate(30)

        rospy.loginfo("Press 'q' to quit")

        while not rospy.is_shutdown():
            with self.lock:
                image = self.current_image
                detections = self.current_detections

            if image is not None:
                # 検出結果を描画
                display_image = self.draw_detections(image.copy(), detections)

                # 検出数を表示
                if detections is not None:
                    num_detections = len(detections.array)
                    cv2.putText(
                        display_image,
                        f"Detections: {num_detections}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

                # 表示
                cv2.imshow("YOLOv5 Viewer", display_image)

            # キー入力チェック
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            rate.sleep()

        cv2.destroyAllWindows()


def main():
    try:
        viewer = YOLOViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
