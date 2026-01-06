#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint


class ArmCommander(Node):
    def __init__(self):
        super().__init__('arm_commander')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._action_client.wait_for_server()
        self.get_logger().info('Connected to move_action server')

    def move_to_joints(self, positions):
        """
        positions: list of 6 floats for joint_0 through joint_5
        """
        joint_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        
        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        for name, pos in zip(joint_names, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        
        goal.request.goal_constraints.append(constraints)
        
        self.get_logger().info(f'Sending goal: {positions}')
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False
        
        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        self.get_logger().info(f'Result: error_code={result.error_code.val}')
        return result.error_code.val == 1  # 1 = SUCCESS


def main():
    rclpy.init()
    commander = ArmCommander()
    
    # Move to position 1
    commander.move_to_joints([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
    
    # Move to position 2
    commander.move_to_joints([0.5, 0.3, 0.8, 0.0, 0.2, 0.0])
    
    # Move home
    commander.move_to_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()