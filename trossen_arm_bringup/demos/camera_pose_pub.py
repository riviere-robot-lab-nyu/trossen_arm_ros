import rclpy
from tf2_ros import TransformListener, Buffer, TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class CameraPosePublisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('camera_pose_publisher')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.1, self.publish_camera_pose)
        
    def publish_camera_pose(self):
        try:
            # Get the transform from camera to tag
            transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame',
                'object',
                rclpy.time.Time()
            )
            
            # Invert it: now it's tag to camera
            inverted = TransformStamped()
            inverted.header.stamp = transform.header.stamp
            inverted.header.frame_id = 'object'
            inverted.child_frame_id = 'camera_pose'
            
            # Invert translation and rotation
            from tf_transformations import quaternion_inverse, quaternion_multiply
            import numpy as np
            
            # Position: -R^T * t
            t = np.array([transform.transform.translation.x,
                         transform.transform.translation.y,
                         transform.transform.translation.z])
            q = [transform.transform.rotation.x,
                 transform.transform.rotation.y,
                 transform.transform.rotation.z,
                 transform.transform.rotation.w]
            
            q_inv = quaternion_inverse(q)
            t_inv = -np.dot(tf_transformations.quaternion_matrix(q_inv)[:3, :3], t)
            
            inverted.transform.translation.x = float(t_inv[0])
            inverted.transform.translation.y = float(t_inv[1])
            inverted.transform.translation.z = float(t_inv[2])
            inverted.transform.rotation.x = q_inv[0]
            inverted.transform.rotation.y = q_inv[1]
            inverted.transform.rotation.z = q_inv[2]
            inverted.transform.rotation.w = q_inv[3]
            
            self.tf_broadcaster.sendTransform(inverted)
            
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")

def main():
    rclpy.init()
    node = CameraPosePublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()