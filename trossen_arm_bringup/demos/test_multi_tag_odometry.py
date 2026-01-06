from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from apriltag_msgs.msg import AprilTagDetectionArray
from sensor_msgs.msg import CameraInfo
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import Odometry
from odom_utils import OdometryEstimator, OdometryState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from threading import Lock


@dataclass
class TagConfig:
    frame_id: str
    translation_from_world: np.ndarray  # tag_3_world → this tag
    rotation_from_world: R              # tag_3_world → this tag
    priority: int = 0                   # Lower = preferred
    # Map from tag ID to frame_id used in TagConfig, tag 3 is the world origin
    tag_id_to_frame = {
        3: 'tag_3_world',
        4: 'tag_4',
        5: 'tag_5',
        6: 'tag_6',
        7: 'tag_7',
        8: 'tag_8',
    }

class PnPPoseEstimator(Node):
    def __init__(self):
        super().__init__('multi_apriltag_pose_estimator')
        
        # Camera intrinsics (from CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        # Known tag 3D corners in world frame
        # Tag size (e.g., 5cm = 0.05m)
        self.tag_size = 0.2
        
        # Known tag positions in world frame (center of each tag)
        self.tag_positions = {
            3: np.array([0.0, 0.0, 0.0]),
            4: np.array([-0.4, -0.02, 0.13]),  # 40cm to the right
        }
        
        # Tag orientations in world frame (if tags are coplanar, all identity)
        self.tag_rotations = {
            3: R.from_quat([0, 0, 0, 1]),
            4: R.from_quat([-0.077, 0.13, 0, 0.988]),  
        }
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detection_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_profile_pub)

        # Timer for publishing world origin in base_link
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1/10, self.publish_camera_odometry) # 10 Hz

        # Track currently visible tags
        self.visible_tags = set()
        self.visible_tags_lock = Lock()
        self.detection_timeout = 0.5  # seconds
        self.last_detection_time = {}
        
        # Store initial pose
        self.initial_position = None
        self.initial_orientation = None
        self.initialized = False
        self.latest_pose = None

        self.odometry = OdometryEstimator(window_size=5, 
                                          use_global_pose=True, 
                                          max_velocity=2.0,        # m/s — comfortable margin above walking speed
                                          max_angular_velocity=2.0  # rad/s — allows reasonably quick turns
                                          )

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
                translation_from_world=np.array([-0.077, 0.13, 0, 0.988]),
                rotation_from_world=R.from_quat([0, 0, 0, 1]),
                priority=1
            ),
        ]

        # Sort by priority
        self.tags.sort(key=lambda t: t.priority)
        
    
    def publish_camera_odometry(self):
        """Publish the camera optical frame in base_link frame as Odometry."""
        try:
            if self.latest_pose is None:
                self.get_logger().warn("No latest pose yet")
                return

            pos, quat_optical_frame, msg_header = self.latest_pose
            camera_link_frame = quat_optical_frame * R.from_euler('zx', [90,90], degrees=True)
            quat_camera_link = camera_link_frame.as_quat(canonical=True)

            if not self.initialized:
                self.initial_position = np.array([
                    pos[0],
                    pos[1],
                    pos[2],
                ])
                # [w, x, y, z]
                self.initial_orientation = np.array([
                    quat_camera_link[3],
                    quat_camera_link[0],
                    quat_camera_link[1],
                    quat_camera_link[2],
                ])
                self.odometry.initialize(self.initial_position, self.initial_orientation)
                self.initialized = True
                self.get_logger().info(f"Odometry initialized at position: {self.initial_position}")
            
            timestamp = msg_header.stamp.sec + msg_header.stamp.nanosec * 1e-9
            
            state = self.odometry.update(pos, quat_camera_link, timestamp)
            
            if state is None:
                self.get_logger().info("Odometry state is None, skipping publish.")
                return
            
            # Build message
            msg = Odometry()
            msg.header.stamp = msg_header.stamp
            msg.header.frame_id = "map"
            msg.child_frame_id = "base_link"
            
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
            
            self.get_logger().info(
                f"Published camera in world: pos={pos}, ori={quat_camera_link}"
            )
            # position logging
            self.get_logger().info(f"Published odometry position x: {msg.pose.pose.position.x:.2f}, y: {msg.pose.pose.position.y:.2f}, z: {msg.pose.pose.position.z:.2f}")
            # orientation logging
            # self.get_logger().info(f"Published odometry orientation w: {msg.pose.pose.orientation.w:.2f}, x: {msg.pose.pose.orientation.x:.2f}, y: {msg.pose.pose.orientation.y:.2f}, z: {msg.pose.pose.orientation.z:.2f}")
            # velocity logging
            # self.get_logger().info(f"Published odometry velocity x: {msg.twist.twist.linear.x:.2f}, y: {msg.twist.twist.linear.y:.2f}, z: {msg.twist.twist.linear.z:.2f}")
            # angular velocity logging
            # self.get_logger().info(f"Published odometry angular velocity x: {msg.twist.twist.angular.x:.2f}, y: {msg.twist.twist.angular.y:.2f}, z: {msg.twist.twist.angular.z:.2f}")
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
    
    def get_tag_corners_3d(self, tag_id: int) -> np.ndarray:
        """
        Get 4 corners of a tag in world frame.
        AprilTag corner order: bottom-left, bottom-right, top-right, top-left
        (when looking at the tag)
        """
        if tag_id not in self.tag_positions:
            return None
        
        center = self.tag_positions[tag_id]
        rotation = self.tag_rotations[tag_id]
        half = self.tag_size / 2
        
        # Corners in tag frame (Z points out of tag)
        corners_local = np.array([
            [-half, -half, 0],  # bottom-left
            [ half, -half, 0],  # bottom-right
            [ half,  half, 0],  # top-right
            [-half,  half, 0],  # top-left
        ])
        
        # Transform to world frame
        corners_world = center + rotation.apply(corners_local)
        return corners_world
    
    def detection_callback(self, msg: AprilTagDetectionArray):
        """Estimate camera pose from all detected tags using PnP."""
        
        if self.camera_matrix is None:
            self.get_logger().warn("No camera intrinsics yet")
            return
        
        # Collect 2D-3D correspondences from all detected tags
        points_3d = []
        points_2d = []
        
        for det in msg.detections:
            tag_id = det.id
            
            corners_3d = self.get_tag_corners_3d(tag_id)
            if corners_3d is None:
                continue
            
            # 2D corners from detection
            corners_2d = np.array([
                [det.corners[0].x, det.corners[0].y],
                [det.corners[1].x, det.corners[1].y],
                [det.corners[2].x, det.corners[2].y],
                [det.corners[3].x, det.corners[3].y],
            ])
            
            points_3d.append(corners_3d)
            points_2d.append(corners_2d)
        
        if len(points_3d) == 0:
            return
        
        # Stack all points
        points_3d = np.vstack(points_3d).astype(np.float64)
        points_2d = np.vstack(points_2d).astype(np.float64)
        
        self.get_logger().debug(f"PnP with {len(points_3d)} points from {len(msg.detections)} tags")
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            self.get_logger().warn("PnP failed")
            return
        
        # Optional: Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(
            points_3d,
            points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            rvec,
            tvec
        )
        
        # Convert to rotation matrix
        R_cam_to_world, _ = cv2.Rodrigues(rvec)
        t_cam_to_world = tvec.flatten()
        
        # Camera pose in world frame
        # R_cam_to_world and t_cam_to_world give: world_point = R @ cam_point + t
        # So camera position in world = -R.T @ t
        cam_position_world = -R_cam_to_world.T @ t_cam_to_world
        # this is camera optical frame to world
        cam_rotation_world = R.from_matrix(R_cam_to_world.T)
        
        self.latest_pose = (cam_position_world, cam_rotation_world, msg.header)
        
        # self.get_logger().info(
        #     f"Camera in world: pos={cam_position_world}, Ori={cam_rotation_world.as_quat()}"
        #     f"rpy={cam_rotation_world.as_euler('xyz', degrees=True)}"
        # )
    
    def get_camera_in_base_link(self) -> tuple:
        """Get camera pose in base_link frame."""
        
        if self.latest_pose is None:
            return None
        
        cam_pos_world, cam_rot_world, msg_header = self.latest_pose
        
        # Get camera in base_link (from robot kinematics)
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_color_optical_frame',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
        except:
            self.get_logger().warn("TF lookup failed for camera_color_optical_frame to base_link")
            return None
        
        cam_pos_base = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        ])
        cam_rot_base = R.from_quat([
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ])
        
        # World origin in camera frame
        world_in_cam_pos = -cam_rot_world.inv().apply(cam_pos_world)
        world_in_cam_rot = cam_rot_world.inv()
        
        # World origin in base_link
        world_in_base_pos = cam_pos_base + cam_rot_base.apply(world_in_cam_pos)
        world_in_base_rot = cam_rot_base * world_in_cam_rot
        
        return world_in_base_pos, world_in_base_rot.as_quat()
    
def main():
    rclpy.init()
    node = PnPPoseEstimator()
    rclpy.spin(node)


if __name__ == '__main__':
    main()