#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from control_msgs.action import GripperCommand
import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers import GripperDemoNode  # noqa: I100

class ArmCommander(Node):
    def __init__(self):
        super().__init__('arm_commander')
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._gripper_action_client = ActionClient(
            self,
            GripperCommand,
            action_name='gripper_controller/gripper_cmd',
        )
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')
        
        self._action_client.wait_for_server()
        self._gripper_action_client.wait_for_server()
        self._ik_client.wait_for_service()
        self.get_logger().info('Connected to move_action, gripper_cmd and IK services')
        
        self.joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']

        self.grasp_position = None
        self.grasp_orientation = None
        self.grasp_once = False

        # Publish static transform: link_5 → camera_color_frame
        self.static_broadcaster = StaticTransformBroadcaster(self)
        # self.publish_static_transforms()

        # Timer for listening TFs and binary cmd to grasp or not 
        # self.create_timer(0.1, self.listen_tf) # 10 Hz

    # def publish_static_transforms(self):
    #     t = TransformStamped()
        
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'link_6'
    #     t.child_frame_id = 'camera_link'
        
    #     # TODO: We should calibrate this transform instead of measuring by hand
    #     # Translation (m)
    #     t.transform.translation.x = 0.0538  # 53.8mm
    #     t.transform.translation.y = 0.0
    #     t.transform.translation.z = 0.05    # 50mm
        
    #     # Rotation: 15 deg around Y
    #     # r = R.from_euler('y', 15, degrees=True)
    #     # quat = r.as_quat()  # [x, y, z, w]
    #     quat = [0.000000, 0.130500, 0.000000, 0.991400]
        
        
    #     t.transform.rotation.x = quat[0]
    #     t.transform.rotation.y = quat[1]
    #     t.transform.rotation.z = quat[2]
    #     t.transform.rotation.w = quat[3]
        
    #     self.static_broadcaster.sendTransform(t)
    #     self.get_logger().info("Published static transform: link_6 → camera_link")

    
    def grasp(self):
        # Wait for TF to be ready (with spinning)
        self.get_logger().info("Waiting for TF...")
        
        transform = None
        for _ in range(50):  # Try for ~5 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    'tag_3_world',
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1)
                )
                break
            except Exception as e:
                self.get_logger().debug(f"Waiting: {e}")
        
        if transform is None:
            self.get_logger().error("Failed to get transform")
            return
        
        # Got it!
        tag_position = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])

        tag_rotation = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ])
        
        self.get_logger().info(f"Tag position in base_link: {tag_position}")
        self.get_logger().info(f"Tag orientation in base_link (quat): {tag_rotation.as_quat()}")

        # Positional grasp offset in tag_3_world frame
        grasp_offset_in_tag = np.array([0.0, 0.2, 0.1]) # 20cm above tag, 10cm forward from tag 
        
        # Rotation offset in tag frame
        grasp_rotation_offset =  R.from_euler('xy', [-90, 90], degrees=True) # R.from_euler('y', 90, degrees=True) * R.from_euler('x', -90, degrees=True)
        # grasp_rotation_offset =  R.from_quat([0.6335811, 0, 0.6335811, 0.4440158])  # 180 deg around Z
        if self.grasp_once==False:
            self.get_logger().info("Calculating grasp pose...")
            # Transform offset to base_link frame
            self.grasp_position = tag_position + tag_rotation.apply(grasp_offset_in_tag)
            grasp_rotation = tag_rotation * grasp_rotation_offset
            self.grasp_orientation = grasp_rotation.as_quat()  # [x, y, z, w]
            self.grasp_once = True
        
        self.get_logger().info(f"Grasp position in base_link: {self.grasp_position}")
        self.get_logger().info(f"Grasp orientation in base_link (quat): {self.grasp_orientation}") 

        # Move to grasp pose
        self.move_to_pose(self.grasp_position[0], 
                          self.grasp_position[1], 
                          self.grasp_position[2], 
                          self.grasp_orientation[0], 
                          self.grasp_orientation[1], 
                          self.grasp_orientation[2], 
                          self.grasp_orientation[3])
        return

    def compute_ik(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Compute IK for a given SE3 pose"""
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'arm'
        request.ik_request.avoid_collisions = True
        
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        
        request.ik_request.pose_stamped = pose
        
        future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        result = future.result()
        if result.error_code.val != 1:
            self.get_logger().error(f'IK failed with error code: {result.error_code.val}')
            return None
        
        # Extract joint positions
        joint_positions = []
        for name in self.joint_names:
            idx = result.solution.joint_state.name.index(name)
            joint_positions.append(result.solution.joint_state.position[idx])
        
        self.get_logger().info(f'IK solution: {joint_positions}')
        return joint_positions

    def move_to_joints(self, positions):
        """Move to joint positions"""
        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        for name, pos in zip(self.joint_names, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        
        goal.request.goal_constraints.append(constraints)
        
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False
        
        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        return result.error_code.val == 1

    def move_to_pose(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Move to SE3 pose (computes IK first)"""
        self.get_logger().info(f'Moving to pose: ({x}, {y}, {z}), ({qx}, {qy}, {qz}, {qw})')
        
        joint_positions = self.compute_ik(x, y, z, qx, qy, qz, qw)
        if joint_positions is None:
            return False
        
        return self.move_to_joints(joint_positions)
    
    def move_gripper(self, position, max_effort=5.0):
        """Move gripper to position"""
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        
        future = self._gripper_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            return False
        
        self.get_logger().info('Executing gripper command...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        self.get_logger().info(f'Gripper command result: {result}')
        return True
    



def main():
    rclpy.init()
    commander = ArmCommander()
    gripper_open_position = 0.04  # fully open
    gripper_closed_position = 0.002 # not fully closed to avoid strain
    good_view_pose = (0.076, -0.044, 0.309, 0.0648, 0.122, -0.376, 0.916)
    commander.move_gripper(gripper_open_position)
    commander.move_to_pose(*good_view_pose)
    
    # move to grasp pose
    commander.grasp()
    # close gripper
    # commander.move_gripper(gripper_closed_position)

    # rotate 90 degrees around x axis
    # commander.move_to_pose(*commander.grasp_position, 
    #                       * (R.from_quat(commander.grasp_orientation) * R.from_euler('x', 90, degrees=True)).as_quat())
    
    # # Move back to home
    # commander.move_to_pose(*good_view_pose)

    # # rotate 180 degrees around x axis
    # rotated_grasp_orientation = R.from_quat([*good_view_pose[3:7]]) * R.from_euler('x', 180, degrees=True)
    # commander.move_to_pose(*good_view_pose[:3], *rotated_grasp_orientation.as_quat())

    # # move back to grasp pose
    # commander.grasp()
    # # put thing down 
    # commander.move_gripper(gripper_open_position)
    # # Move back to home
    # commander.move_to_pose(*good_view_pose)
    
    commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()