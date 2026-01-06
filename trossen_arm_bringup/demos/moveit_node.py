from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.constants import S_TO_NS
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

class MoveItNode(Node):
    def __init__(
        self,
        namespace: str = '',
        action_name: str = '/move_group',
    ):
        super().__init__('moveit_node')
        self.joint_names = [
            'joint_0',
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
        ]

        if namespace:
            action_name = f'{namespace}/{action_name}'
            for i in range(len(self.joint_names)):
                self.joint_names[i] = f'{namespace}/{self.joint_names[i]}'

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            action_name=action_name,
        )
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(
                f"Waiting for '{self._action_client._action_name}' action server..."
            )
        self._is_running = False
        self.get_logger().info(f'MoveItNode initialized with action server: {action_name}')