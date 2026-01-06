import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class OdometryState:
    position: np.ndarray
    orientation: np.ndarray  # [x, y, z, w]
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float

class OdometryEstimator:
    def __init__(self, window_size: int = 5, use_global_pose: bool = False, max_velocity: float = 2.0, max_angular_velocity: float = 2.0):
        """
        Args:
            window_size: Number of past measurements to use for velocity estimation.
            use_global_pose: If True, output global ENU pose instead of relative pose.
            max_velocity: Maximum linear velocity (m/s) for sanity check.
            max_angular_velocity: Maximum angular velocity (rad/s) for sanity check.
        """
        self.window_size = window_size
        self.initialized = False
        self.use_global_pose = use_global_pose
        
        # Initial pose (reference frame)
        self.initial_position: Optional[np.ndarray] = None
        self.r_initial: Optional[R] = None
        
        # History for velocity estimation
        self.position_history = deque(maxlen=window_size)
        self.orientation_history = deque(maxlen=window_size)
    
    def initialize(self, position: np.ndarray, orientation: np.ndarray):
        """
        Set initial pose as reference.
        orientation: [w, x, y, z] format from TF
        """
        self.initial_position = position.copy()
        self.r_initial = R.from_quat([
            orientation[1],  # x
            orientation[2],  # y
            orientation[3],  # z
            orientation[0]   # w
        ])
        self.initialized = True
        self.position_history.clear()
        self.orientation_history.clear()
    
    def reset(self):
        """Reset odometry to uninitialized state."""
        self.initialized = False
        self.initial_position = None
        self.r_initial = None
        self.position_history.clear()
        self.orientation_history.clear()
    
    def update(self, position: np.ndarray, orientation: np.ndarray, timestamp: float) -> Optional[OdometryState]:
        """
        Update odometry with new measurement.
        
        Args:
            position: [x, y, z] in world frame
            orientation: [w, x, y, z] from TF
            timestamp: seconds
        
        Returns:
            OdometryState with relative pose and velocities, or None if not initialized
        """
        if not self.initialized:
            self.initialize(position, orientation)
            return OdometryState(
                position=np.zeros(3),
                orientation=np.array([0, 0, 0, 1]),  # identity [x,y,z,w]
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                timestamp=timestamp
            )
        
        # Compute relative position (in camera frame)
        rel_pos_camera = position - self.initial_position
        
        # Convert to robot frame (x forward, y left, z up)
        rel_position = np.array([
            -rel_pos_camera[2],  # robot_x = -camera_z
            -rel_pos_camera[0],  # robot_y = -camera_x
             rel_pos_camera[1]   # robot_z = +camera_y
        ])
        
        # Compute relative orientation
        r_current = R.from_quat([
            orientation[1], orientation[2], orientation[3], orientation[0]
        ])
        # TODO: Verify direction of relative rotation. this match in real world but does it make sense?
        rel_rot = self.r_initial * r_current.inv()
        # rel_rot = r_current * self.r_initial.inv()
        rel_quat_camera = rel_rot.as_quat()  # [x, y, z, w]
        
        # Convert quaternion to robot frame
        rel_orientation = np.array([
            -rel_quat_camera[2],  # robot_qx = -camera_qz
            -rel_quat_camera[0],  # robot_qy = -camera_qx
             rel_quat_camera[1],  # robot_qz = +camera_qy
             rel_quat_camera[3]   # w unchanged
        ])
        
        # Compute velocities
        linear_vel = self._compute_linear_velocity(rel_position, timestamp)
        angular_vel = self._compute_angular_velocity(rel_orientation, timestamp)

        if self.use_global_pose==False:
            return OdometryState(
                position=rel_position,
                orientation=rel_orientation,
                linear_velocity=linear_vel,
                angular_velocity=angular_vel,
                timestamp=timestamp
            )
        else:
            ENU_position = np.array([
                -position[2], # x = -z
                -position[0], # y = -x
                 position[1]  # z = y
            ])
            return OdometryState(
                position=ENU_position,
                orientation=orientation,
                linear_velocity=linear_vel,
                angular_velocity=angular_vel,
                timestamp=timestamp
            )
    
    def _compute_linear_velocity(self, position: np.ndarray, timestamp: float) -> np.ndarray:
        self.position_history.append((timestamp, position.copy()))
        
        if len(self.position_history) < 2:
            return np.zeros(3)
        
        oldest_time, oldest_pos = self.position_history[0]
        newest_time, newest_pos = self.position_history[-1]
        
        dt = newest_time - oldest_time
        if dt < 1e-6:
            return np.zeros(3)
        
        return (newest_pos - oldest_pos) / dt
    
    def _compute_angular_velocity(self, orientation: np.ndarray, timestamp: float) -> np.ndarray:
        """orientation: [x, y, z, w] format"""
        self.orientation_history.append((timestamp, orientation.copy()))
        
        if len(self.orientation_history) < 2:
            return np.zeros(3)
        
        oldest_time, oldest_quat = self.orientation_history[0]
        newest_time, newest_quat = self.orientation_history[-1]
        
        dt = newest_time - oldest_time
        if dt < 1e-6:
            return np.zeros(3)
        
        r_old = R.from_quat(oldest_quat)
        r_new = R.from_quat(newest_quat)
        r_rel = r_old * r_new.inv()  # Body frame
        
        return r_rel.as_rotvec() / dt