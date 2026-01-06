import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from rclpy.duration import Duration
import numpy as np
from collections import deque

class AprilTagOdometryPublisher(Node):
    def __init__(self):
        super().__init__('apriltag_odometry_publisher')
        
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, '/odometry/apriltag', 10)
        
        # Timer for publishing
        self.create_timer(0.1, self.publish_apriltag_odometry) # 10 Hz

        # Sliding window for velocity computation
        self.window_size = 5  # Number of samples
        self.position_history = deque(maxlen=self.window_size)
        self.orientation_history = deque(maxlen=self.window_size-3)
        
        # Store initial pose
        self.initial_position = None
        self.initial_orientation = None
        self.initialized = False

    def compute_velocity(self, current_position, current_time_sec):
        """Compute velocity using sliding window linear regression."""
        self.position_history.append((current_time_sec, current_position.copy()))
        
        if len(self.position_history) < 2:
            return np.zeros(3)
        
        # Simple approach: (newest - oldest) / dt
        oldest_time, oldest_pos = self.position_history[0]
        newest_time, newest_pos = self.position_history[-1]
        
        dt = newest_time - oldest_time
        if dt < 1e-6:
            return np.zeros(3)
        
        velocity = (newest_pos - oldest_pos) / dt
        return velocity
    
    def angular_velocities(self, q1, q2, dt):
        return (2 / dt) * np.array([
            q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
            q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
            q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])
    
    def compute_angular_velocity(self, current_quat, current_time_sec):
        """Compute angular velocity using sliding window.
        
        current_quat: [x, y, z, w] format
        """
        self.orientation_history.append((current_time_sec, current_quat.copy()))
        
        if len(self.orientation_history) < 2:
            return np.zeros(3)
        
        oldest_time, oldest_quat = self.orientation_history[0]
        newest_time, newest_quat = self.orientation_history[-1]
        
        dt = newest_time - oldest_time
        if dt < 1e-6:
            return np.zeros(3)
        
        # Compute relative rotation: q_rel = q_new * q_old^-1
        r_old = R.from_quat(oldest_quat)  # [x, y, z, w]
        r_new = R.from_quat(newest_quat)
        r_rel = r_new * r_old.inv()
        
        # Convert to axis-angle (rotvec = axis * angle)
        rotvec = r_rel.as_rotvec()
        
        # Angular velocity = rotation / dt
        angular_velocity = rotvec / dt
        # angular_velocity = self.angular_velocities(oldest_quat, newest_quat, dt)
        return angular_velocity
        
    def publish_apriltag_odometry(self):
        try:
            # set object as world frame
            # camera in world frame
            transform_cam_w = self.tf_buffer.lookup_transform(
                'tag_3_world',
                'camera_color_frame',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)  # Add timeout (shorter than 0.1s timer)
            )
            
            # Translation from tag4 to camera
            transform_tag4_w = self.tf_buffer.lookup_transform(
                'tag_3_world',
                'tag_4',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)  # Add timeout (shorter than 0.1s timer)
            )
            # using transform_tag4_w will be less accurate than our preset known transform
            # translation_tag3_to_tag4 = np.array([0.33, 0.03, -0.12])  # Default zero


            # Initialize on first detection
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

                self.r_initial = R.from_quat([
                    self.initial_orientation[1],
                    self.initial_orientation[2],
                    self.initial_orientation[3],
                    self.initial_orientation[0]
                ])
                self.initialized = True
                self.get_logger().info(f"Odometry initialized at position: {self.initial_position}")
            
            self.get_logger().info("Past initialization block")
            msg = Odometry()
            msg.header.frame_id = "odom"  # Odometry frame (relative to start)
            msg.child_frame_id = "camera_color_frame"  # Robot/camera frame
            msg.header.stamp =  transform_cam_w.header.stamp
            
            # Position relative to initial position
            relative_position = np.array([
                -(transform_cam_w.transform.translation.z - self.initial_position[2]),
                -(transform_cam_w.transform.translation.x - self.initial_position[0]),
                transform_cam_w.transform.translation.y - self.initial_position[1]
            ])
            self.last_position = relative_position


            # Orientation relative to initial orientation
            current_orientation = np.array([
                transform_cam_w.transform.rotation.w,
                transform_cam_w.transform.rotation.x,
                transform_cam_w.transform.rotation.y,
                transform_cam_w.transform.rotation.z
            ])
            
            r_current = R.from_quat([
                current_orientation[1], 
                current_orientation[2],
                current_orientation[3],
                current_orientation[0]
            ])

            # Relative orientation
            relative_rot = self.r_initial * r_current.inv()
            # relative_rot = r_current * self.r_initial.inv()
            relative_quat = relative_rot.as_quat()  # Returns [x, y, z, w]

            # Convert to robot frame convention (same remapping as position)
            converted_quat = np.array([
                -relative_quat[2],  # robot_qx ← -camera_qz
                -relative_quat[0],  # robot_qy ← -camera_qx
                relative_quat[1],   # robot_qz ← +camera_qy
                relative_quat[3]    # w unchanged
            ])
            
            msg.pose.pose.position.x = relative_position[0]
            msg.pose.pose.position.y = relative_position[1]
            msg.pose.pose.position.z = relative_position[2]
            
            # Orientation (absolute orientation from tag)
            msg.pose.pose.orientation.w = converted_quat[3]
            msg.pose.pose.orientation.x = converted_quat[0]
            msg.pose.pose.orientation.y = converted_quat[1]
            msg.pose.pose.orientation.z = converted_quat[2]

            current_time_sec =  transform_cam_w.header.stamp.sec + transform_cam_w.header.stamp.nanosec * 1e-9
            velocity = self.compute_velocity(relative_position, current_time_sec)

            # Angular velocity - pass quaternion in [x, y, z, w] format
            angular_velocity = self.compute_angular_velocity(converted_quat, current_time_sec)  
            # Velocity (dummy values for now)
            msg.twist.twist.linear.x = velocity[0]
            msg.twist.twist.linear.y = velocity[1]
            msg.twist.twist.linear.z = velocity[2]
            msg.twist.twist.angular.x = angular_velocity[0]
            msg.twist.twist.angular.y = angular_velocity[1]
            msg.twist.twist.angular.z = angular_velocity[2]

            self.odom_pub.publish(msg)
            # position logging
            self.get_logger().info(f"Published odometry position x: {msg.pose.pose.position.x:.2f}, y: {msg.pose.pose.position.y:.2f}, z: {msg.pose.pose.position.z:.2f}")
            # orientation logging
            self.get_logger().info(f"Published odometry orientation w: {msg.pose.pose.orientation.w:.2f}, x: {msg.pose.pose.orientation.x:.2f}, y: {msg.pose.pose.orientation.y:.2f}, z: {msg.pose.pose.orientation.z:.2f}")
            # velocity logging
            # self.get_logger().info(f"Published odometry velocity x: {msg.twist.twist.linear.x:.2f}, y: {msg.twist.twist.linear.y:.2f}, z: {msg.twist.twist.linear.z:.2f}")
            # angular velocity logging
            # self.get_logger().info(f"Published odometry angular velocity x: {msg.twist.twist.angular.x:.2f}, y: {msg.twist.twist.angular.y:.2f}, z: {msg.twist.twist.angular.z:.2f}")
            # tag4 position logging
            # self.get_logger().info(f"Tag4 position in world frame x: {transform_tag4_w.transform.translation.x:.2f}, y: {transform_tag4_w.transform.translation.y:.2f}, z: {transform_tag4_w.transform.translation.z:.2f}")
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")


def main():
    rclpy.init()
    node = AprilTagOdometryPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()