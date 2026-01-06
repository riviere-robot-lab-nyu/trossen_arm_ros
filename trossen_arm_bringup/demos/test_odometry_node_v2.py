from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.duration import Duration

from apriltag_msgs.msg import AprilTagDetectionArray
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import Odometry
from odom_utils import OdometryEstimator, OdometryState

@dataclass
class TagConfig:
    frame_id: str
    translation_from_world: np.ndarray  # tag_3_world → this tag
    rotation_from_world: R              # tag_3_world → this tag
    priority: int = 0                   # Lower = preferred
    # Map from tag ID to frame_id used in TagConfig
    tag_id_to_frame = {
        3: 'tag_3_world',
        4: 'tag_4',
        5: 'tag_5',
        6: 'tag_6',
        7: 'tag_7',
        8: 'tag_8',
    }

class AprilTagOdometryPublisher(Node):
    def __init__(self):
        super().__init__('apriltag_odometry_publisher')
        
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, '/odometry/apriltag_v2', 10)
        
        # Subscriber 
        # Detections
        self.detection_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detection_callback,
            10
        )
        # Timer for publishing
        self.create_timer(0.1, self.publish_apriltag_odometry) # 10 Hz

        # Track currently visible tags
        self.visible_tags = set()
        self.visible_tags_lock = Lock()
        self.detection_timeout = 0.5  # seconds
        self.last_detection_time = {}
        
        # Store initial pose
        self.initial_position = None
        self.initial_orientation = None
        self.initialized = False

        self.odometry = OdometryEstimator()

        # Map from tag ID to frame_id used in TagConfig
        self.tag_id_to_frame = TagConfig.tag_id_to_frame

        # Tag configurations (all transforms relative to tag_3_world)
        self.tags = [
            TagConfig(
                frame_id='tag_3_world',
                translation_from_world=np.array([0.0, 0.0, 0.0]),
                rotation_from_world=R.from_quat([0, 0, 0, 1]),
                priority=0  # Primary tag
            ),
            TagConfig(
                frame_id='tag_4',
                translation_from_world=np.array([0.40, 0.02, -0.05]),
                rotation_from_world=R.from_quat([0, 0, 0, 1]),
                priority=1
            ),
            # TagConfig(
            #     frame_id='tag_5',
            #     translation_from_world=np.array([0.66, 0.0, 0.0]),
            #     rotation_from_world=R.from_quat([0, 0, 0, 1]),
            #     priority=2
            # ),
            # TagConfig(
            #     frame_id='tag_6',
            #     translation_from_world=np.array([1.0, 0.0, 0.0]),
            #     rotation_from_world=R.from_quat([0, 0, 0, 1]),
            #     priority=3
            # ),
            # TagConfig(
            #     frame_id='tag_7',
            #     translation_from_world=np.array([-0.33, 0.0, 0.0]),
            #     rotation_from_world=R.from_quat([0, 0, 0, 1]),
            #     priority=4
            # ),
            # TagConfig(
            #     frame_id='tag_8',
            #     translation_from_world=np.array([-0.66, 0.0, 0.0]),
            #     rotation_from_world=R.from_quat([0, 0, 0, 1]),
            #     priority=5
            # ),
        ]
        
        # Sort by priority
        self.tags.sort(key=lambda t: t.priority)

    def detection_callback(self, msg: AprilTagDetectionArray):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        
        with self.visible_tags_lock:
            # Update visible tags from this detection
            for detection in msg.detections:
                tag_id = detection.id
                if tag_id in self.tag_id_to_frame:
                    frame_id = self.tag_id_to_frame[tag_id]
                    self.visible_tags.add(frame_id)
                    self.last_detection_time[frame_id] = current_time
            
            # Remove stale tags
            stale = [
                frame_id for frame_id, t in self.last_detection_time.items()
                if current_time - t > self.detection_timeout
            ]
            for frame_id in stale:
                self.visible_tags.discard(frame_id)
                del self.last_detection_time[frame_id]

    def publish_apriltag_odometry(self):
        """Publish odometry based on AprilTag detections."""
        try:            # Get camera transform in world frame
            transform_cam_w, used_tag_frame = self.get_camera_transform()
            if transform_cam_w is None:
                self.get_logger().info("No valid tag transform found.")
                return
            
            if not self.initialized:
                self.initial_position = np.array([
                    transform_cam_w.transform.translation.x,
                    transform_cam_w.transform.translation.y,
                    transform_cam_w.transform.translation.z
                ])
                self.initial_orientation = np.array([
                    transform_cam_w.transform.rotation.w,
                    transform_cam_w.transform.rotation.x,
                    transform_cam_w.transform.rotation.y,
                    transform_cam_w.transform.rotation.z
                ])
                self.odometry.initialize(self.initial_position, self.initial_orientation)
                self.initialized = True
                self.get_logger().info(f"Odometry initialized at position: {self.initial_position}")

            self.get_logger().info(f"Using tag frame: {used_tag_frame}")
            position = np.array([
                transform_cam_w.transform.translation.x,
                transform_cam_w.transform.translation.y,
                transform_cam_w.transform.translation.z
            ])
            orientation = np.array([
                transform_cam_w.transform.rotation.w,
                transform_cam_w.transform.rotation.x,
                transform_cam_w.transform.rotation.y,
                transform_cam_w.transform.rotation.z
            ])
            timestamp = transform_cam_w.header.stamp.sec + transform_cam_w.header.stamp.nanosec * 1e-9
            
            state = self.odometry.update(position, orientation, timestamp)
            
            if state is None:
                self.get_logger().info("Odometry state is None, skipping publish.")
                return
            
            # Build message
            msg = Odometry()
            msg.header.stamp = transform_cam_w.header.stamp
            msg.header.frame_id = "odom"
            msg.child_frame_id = "camera_color_frame"
            
            msg.pose.pose.position.x = state.position[0]
            msg.pose.pose.position.y = state.position[1]
            msg.pose.pose.position.z = state.position[2]
            
            msg.pose.pose.orientation.x = state.orientation[0]
            msg.pose.pose.orientation.y = state.orientation[1]
            msg.pose.pose.orientation.z = state.orientation[2]
            msg.pose.pose.orientation.w = state.orientation[3]
            
            msg.twist.twist.linear.x = state.linear_velocity[0]
            msg.twist.twist.linear.y = state.linear_velocity[1]
            msg.twist.twist.linear.z = state.linear_velocity[2]
            
            msg.twist.twist.angular.x = state.angular_velocity[0]
            msg.twist.twist.angular.y = state.angular_velocity[1]
            msg.twist.twist.angular.z = state.angular_velocity[2]
            
            self.odom_pub.publish(msg)

            # position logging
            # self.get_logger().info(f"Published odometry position x: {msg.pose.pose.position.x:.2f}, y: {msg.pose.pose.position.y:.2f}, z: {msg.pose.pose.position.z:.2f}")
            # orientation logging
            self.get_logger().info(f"Published odometry orientation w: {msg.pose.pose.orientation.w:.2f}, x: {msg.pose.pose.orientation.x:.2f}, y: {msg.pose.pose.orientation.y:.2f}, z: {msg.pose.pose.orientation.z:.2f}")
            # velocity logging
            # self.get_logger().info(f"Published odometry velocity x: {msg.twist.twist.linear.x:.2f}, y: {msg.twist.twist.linear.y:.2f}, z: {msg.twist.twist.linear.z:.2f}")
            # angular velocity logging
            self.get_logger().info(f"Published odometry angular velocity x: {msg.twist.twist.angular.x:.2f}, y: {msg.twist.twist.angular.y:.2f}, z: {msg.twist.twist.angular.z:.2f}")
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            
            

    def get_camera_transform(self) -> Tuple[Optional[any], Optional[str]]:
        """Get camera transform in world frame, trying tags by priority."""

        with self.visible_tags_lock:
            visible = self.visible_tags.copy()
        
        for tag in self.tags:  # Already sorted by priority
            if tag.frame_id not in visible:
                continue  # Skip tags that aren't currently detected
            try:
                # Get tag → camera transform
                transform_tag_to_cam = self.tf_buffer.lookup_transform(
                    tag.frame_id,
                    'camera_color_frame',
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.01)
                )
                
                # Extract transform components
                r_tag_to_cam = R.from_quat([
                    transform_tag_to_cam.transform.rotation.x,
                    transform_tag_to_cam.transform.rotation.y,
                    transform_tag_to_cam.transform.rotation.z,
                    transform_tag_to_cam.transform.rotation.w
                ])
                t_tag_to_cam = np.array([
                    transform_tag_to_cam.transform.translation.x,
                    transform_tag_to_cam.transform.translation.y,
                    transform_tag_to_cam.transform.translation.z
                ])
                
                # Chain: world → tag → camera
                r_world_to_cam = tag.rotation_from_world * r_tag_to_cam
                t_world_to_cam = tag.translation_from_world + tag.rotation_from_world.apply(t_tag_to_cam)
                
                # Build transform message
                transform_tag_to_cam.transform.translation.x = t_world_to_cam[0]
                transform_tag_to_cam.transform.translation.y = t_world_to_cam[1]
                transform_tag_to_cam.transform.translation.z = t_world_to_cam[2]
                
                quat = r_world_to_cam.as_quat()
                transform_tag_to_cam.transform.rotation.x = quat[0]
                transform_tag_to_cam.transform.rotation.y = quat[1]
                transform_tag_to_cam.transform.rotation.z = quat[2]
                transform_tag_to_cam.transform.rotation.w = quat[3]
                
                return transform_tag_to_cam, tag.frame_id
                
            except Exception:
                continue
        
        return None, None
    
def main():
    rclpy.init()
    node = AprilTagOdometryPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()