#!/usr/bin/env python3
"""
Video File Publisher for ROS
Reads video file and publishes frames as ROS Image messages.
"""
import os
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class VideoPublisher:
    """Publishes video frames to ROS topic."""

    def __init__(self):
        rospy.init_node('video_publisher', anonymous=False)

        # Get parameters from ROS params or environment variables
        self.video_path = rospy.get_param(
            '~video_file',
            os.environ.get('VIDEO_FILE', '/videos/video.mp4')
        )
        self.fps = rospy.get_param(
            '~fps',
            int(os.environ.get('VIDEO_FPS', '30'))
        )
        self.loop = rospy.get_param(
            '~loop',
            os.environ.get('VIDEO_LOOP', 'true').lower() == 'true'
        )
        self.topic = rospy.get_param(
            '~topic',
            '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        )

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.topic, Image, queue_size=1)

        rospy.loginfo(f"=== Video Publisher Starting ===")
        rospy.loginfo(f"Video file: {self.video_path}")
        rospy.loginfo(f"Target FPS: {self.fps}")
        rospy.loginfo(f"Loop: {self.loop}")
        rospy.loginfo(f"Publishing to: {self.topic}")

    def run(self):
        """Main loop to read and publish video frames."""
        rate = rospy.Rate(self.fps)

        while not rospy.is_shutdown():
            # Open video file
            cap = cv2.VideoCapture(self.video_path)

            if not cap.isOpened():
                rospy.logerr(f"Failed to open video file: {self.video_path}")
                rospy.sleep(1.0)
                continue

            # Get video info
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            rospy.loginfo(f"Video opened: {frame_count} frames at {video_fps:.1f} fps")

            frame_num = 0
            while cap.isOpened() and not rospy.is_shutdown():
                ret, frame = cap.read()

                if not ret:
                    rospy.loginfo("End of video reached")
                    break

                try:
                    # Convert to ROS message
                    msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = 'head_rgbd_sensor_rgb_frame'
                    self.pub.publish(msg)
                    frame_num += 1
                except Exception as e:
                    rospy.logerr(f"Failed to publish frame: {e}")

                rate.sleep()

            cap.release()
            rospy.loginfo(f"Published {frame_num} frames")

            if not self.loop:
                rospy.loginfo("Loop disabled, exiting")
                break

            rospy.loginfo("Restarting video...")

        rospy.loginfo("Video publisher shutting down")


if __name__ == '__main__':
    try:
        publisher = VideoPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
