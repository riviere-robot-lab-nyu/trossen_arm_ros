#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from control_msgs.action import GripperCommand
from controllers import GripperDemoNode  # noqa: I100

class ArmCommander(Node):
    def __init__(self):
        super().__init__('arm_commander')
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
    
    def move_gripper(self, position, max_effort=10.0):
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
    gripper_closed_position = 0.0
    
    # Move to SE3 pose (x, y, z, qx, qy, qz, qw)
    commander.move_to_pose(0.25, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)
    # Open gripper
    # commander.move_gripper(gripper_open_position)
    # Close gripper
    # commander.move_gripper(gripper_closed_position)
    
    # Another pose
    commander.move_to_pose(0.25, 0.0, 0.2, 0.0, 0.0, 0.707, 0.707)

    # Move back to home
    # commander.move_to_pose(0.25, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)
    
    commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()