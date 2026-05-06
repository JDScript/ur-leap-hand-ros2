#!/usr/bin/env python3
"""
Hand-eye calibration collector and solver.
Eye-to-hand: camera fixed, AprilTag on EEF.
Moves robot through predefined joint poses via MoveIt, detects AprilTag,
solves for base->camera transform using cv2.calibrateHandEye.
"""

import os
import sys
import time
import threading
import numpy as np
import cv2
import yaml
from pathlib import Path
from packaging import version
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from rclpy.time import Time
from ament_index_python.packages import get_package_share_directory

import tf2_ros

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from cv_bridge import CvBridge


class HandEyeCollector(Node):
    def __init__(self):
        super().__init__('hand_eye_collector')
        self.declare_parameter('move_group', 'ur_arm')

        # Load config
        pkg_share = Path(get_package_share_directory('hand_eye_calibration'))
        config_path = pkg_share / 'config' / 'calibration_poses.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.joint_names = cfg['joint_names']
        self.poses = cfg['poses']
        self.marker_size = cfg['marker_size']
        self.marker_id = cfg.get('marker_id', -1)

        # Output path
        self.output_path = pkg_share / 'config' / 'camera_base_tf.yaml'

        # ArUco setup — handles both OpenCV <4.7 and >=4.7 APIs
        self._cv2_new_api = version.parse(cv2.__version__) >= version.parse('4.7.0')
        if self._cv2_new_api:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        else:
            self._aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36H11)
            self._aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None
        self.get_logger().info(f'OpenCV {cv2.__version__}: using {"new" if self._cv2_new_api else "legacy"} ArUco API')

        # State
        self.bridge = CvBridge()
        self.latest_image = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_tcp_pose = None
        self.image_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        # TF buffer for static transform lookup (d435_link → d435_color_optical_frame)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveIt action client
        self.moveit_client = ActionClient(self, MoveGroup, '/move_action')

        # Subscribers
        self.create_subscription(Image, '/camera/d435/color/image_raw',
                                 self._image_cb, 10)
        self.create_subscription(CameraInfo, '/camera/d435/color/camera_info',
                                 self._camera_info_cb, 5)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose',
                                 self._tcp_cb, 10)

        self.get_logger().info('HandEyeCollector initialized')
        self.get_logger().info(f'Loaded {len(self.poses)} calibration poses')

    def _image_cb(self, msg):
        with self.image_lock:
            self.latest_image = msg

    def _camera_info_cb(self, msg):
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
            self.camera_matrix = K
            self.dist_coeffs = D
            self.get_logger().info(f'Camera intrinsics loaded: fx={K[0,0]:.1f}')

    def _tcp_cb(self, msg):
        with self.pose_lock:
            self.latest_tcp_pose = msg.pose

    def _pose_to_matrix(self, pose):
        """geometry_msgs/Pose -> 4x4 numpy matrix"""
        T = np.eye(4)
        q = [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def _move_to_joints(self, joint_angles):
        """Send joint space goal to MoveIt, return True on success."""
        if not self.moveit_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveIt action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = self.get_parameter('move_group').value
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        constraints = Constraints()
        for name, angle in zip(self.joint_names, joint_angles):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(angle)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        future = self.moveit_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if not future.done() or not future.result().accepted:
            self.get_logger().error('Goal rejected or timed out')
            return False

        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

        if not result_future.done():
            self.get_logger().error('Motion timed out')
            return False

        error_code = result_future.result().result.error_code.val
        if error_code == 1:
            return True
        else:
            self.get_logger().warn(f'MoveIt error code: {error_code}')
            return False

    def _draw_and_show(self, img, corners, ids, rvecs=None, tvecs=None, label=''):
        """Draw AprilTag detection overlay and show in OpenCV window."""
        vis = img.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if rvecs is not None and tvecs is not None:
                for rvec, tvec in zip(rvecs, tvecs):
                    try:
                        if self._cv2_new_api:
                            cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs,
                                              rvec, tvec, self.marker_size * 0.5)
                        else:
                            cv2.aruco.drawAxis(vis, self.camera_matrix, self.dist_coeffs,
                                               rvec, tvec, self.marker_size * 0.5)
                    except Exception:
                        pass
            color = (0, 255, 0)
            det_text = f'DETECTED  id={ids.flatten()[0]}'
        else:
            color = (0, 80, 255)
            det_text = 'NOT DETECTED'
        cv2.putText(vis, det_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if label:
            cv2.putText(vis, label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow('AprilTag Calibration', vis)
        cv2.waitKey(1)

    def _settle_and_preview(self, duration, label=''):
        """Wait for robot to settle while continuously showing live detection preview."""
        settle_end = time.time() + duration
        while time.time() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.camera_matrix is None:
                continue
            with self.image_lock:
                img_msg = self.latest_image
            if img_msg is None:
                continue
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self._cv2_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=self._aruco_params)
            rvecs = tvecs = None
            if ids is not None and len(ids) > 0:
                try:
                    rv, tv, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[:1], self.marker_size, self.camera_matrix, self.dist_coeffs)
                    rvecs, tvecs = rv[:, 0], tv[:, 0]
                except Exception:
                    pass
            self._draw_and_show(img, corners, ids, rvecs, tvecs, label=label)

    def _detect_tag(self, num_frames=10, label=''):
        """
        Average tag detection over multiple frames for stability.
        Returns T_cam_tag (4x4) or None if not detected.
        """
        if self.camera_matrix is None:
            self.get_logger().error('Camera intrinsics not yet received')
            return None

        rvecs_list = []
        tvecs_list = []
        deadline = time.time() + 5.0  # 5s timeout
        last_img_stamp = None

        while len(rvecs_list) < num_frames and time.time() < deadline:
            with self.image_lock:
                img_msg = self.latest_image

            if img_msg is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

            # Skip if same frame as last iteration
            img_stamp = (img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)
            if img_stamp == last_img_stamp:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue
            last_img_stamp = img_stamp

            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self._cv2_new_api:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=self._aruco_params)

            # Visualize every frame (detected or not)
            vis_rvecs = vis_tvecs = None
            if ids is not None and len(ids) > 0:
                try:
                    rv, tv, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[:1], self.marker_size, self.camera_matrix, self.dist_coeffs)
                    vis_rvecs, vis_tvecs = rv[:, 0], tv[:, 0]
                except Exception:
                    pass
            self._draw_and_show(img, corners, ids, vis_rvecs, vis_tvecs,
                                label=f'{label} | {len(rvecs_list)}/{num_frames} frames')

            if ids is None:
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

            # Filter by marker_id if specified
            if self.marker_id >= 0:
                mask = (ids.flatten() == self.marker_id)
                if not np.any(mask):
                    rclpy.spin_once(self, timeout_sec=0.05)
                    continue
                corners = [corners[i] for i in range(len(ids)) if mask[i]]

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[:1], self.marker_size,
                self.camera_matrix, self.dist_coeffs)

            rvecs_list.append(rvecs[0][0])
            tvecs_list.append(tvecs[0][0])
            rclpy.spin_once(self, timeout_sec=0.05)

        if len(rvecs_list) < 3:
            self.get_logger().warn(f'Only {len(rvecs_list)} frames detected, need at least 3')
            return None

        # Average translation
        t_avg = np.mean(tvecs_list, axis=0)

        # Average rotation via quaternion averaging
        quats = []
        for rvec in rvecs_list:
            R, _ = cv2.Rodrigues(rvec)
            q = Rotation.from_matrix(R).as_quat()
            quats.append(q)
        quats = np.array(quats)
        R_avg = Rotation.from_quat(quats).mean().as_matrix()

        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_avg
        T_cam_tag[:3, 3] = t_avg
        return T_cam_tag

    def _lookup_link_to_optical(self):
        """
        Look up T_{d435_link → d435_color_optical_frame} from the TF tree.
        Returns a 4x4 numpy matrix, or None if unavailable.
        """
        deadline = time.time() + 5.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    'd435_color_optical_frame', 'd435_link', Time())
                t = tf_stamped.transform.translation
                r = tf_stamped.transform.rotation
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
                T[:3, 3] = [t.x, t.y, t.z]
                self.get_logger().info(
                    f'Got d435_link→optical TF: t=[{t.x:.4f},{t.y:.4f},{t.z:.4f}]')
                return T
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass
        self.get_logger().warn(
            'Could not look up d435_link→d435_color_optical_frame. '
            'Is the RealSense driver running? Saving raw calibration without TF correction.')
        return None

    def _solve_calibration(self, samples):
        """
        Solve eye-to-hand calibration from samples list of (T_base_eef, T_cam_tag).
        For eye-to-hand: pass inverted FK poses to calibrateHandEye.
        Returns (R_cam2base, t_cam2base) with lowest consistency error.
        """
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for T_base_eef, T_cam_tag in samples:
            # Eye-to-hand: invert FK pose
            T_eef_base = np.linalg.inv(T_base_eef)
            R_gripper2base.append(T_eef_base[:3, :3])
            t_gripper2base.append(T_eef_base[:3, 3:4])
            R_target2cam.append(T_cam_tag[:3, :3])
            t_target2cam.append(T_cam_tag[:3, 3:4])

        methods = {
            'TSAI': cv2.CALIB_HAND_EYE_TSAI,
            'PARK': cv2.CALIB_HAND_EYE_PARK,
            'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
            'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        }

        best_method = None
        best_error = float('inf')
        best_R = None
        best_t = None

        for name, method in methods.items():
            try:
                R, t = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method)

                # Consistency check: T_eef_tag = inv(T_base_eef) @ T_cam2base @ T_cam_tag
                # should be constant across all samples
                T_cam2base = np.eye(4)
                T_cam2base[:3, :3] = R
                T_cam2base[:3, 3] = t.flatten()

                errors = []
                for T_b_e, T_c_t in samples:
                    # T_eef_tag = (eef_T_base) @ (base_T_cam via cam2base) @ (cam_T_tag)
                    # should be constant across all samples
                    T_eef_tag = np.linalg.inv(T_b_e) @ T_cam2base @ T_c_t
                    errors.append(T_eef_tag)

                tvecs_eef_tag = np.array([e[:3, 3] for e in errors])
                error = np.std(tvecs_eef_tag, axis=0).mean()

                self.get_logger().info(f'{name}: consistency error = {error*1000:.2f} mm')

                if error < best_error:
                    best_error = error
                    best_method = name
                    best_R = R
                    best_t = t
            except Exception as e:
                self.get_logger().warn(f'{name} failed: {e}')

        self.get_logger().info(f'Best method: {best_method} (error={best_error*1000:.2f}mm)')
        return best_R, best_t.flatten()

    def _save_result(self, R_cam2base, t_cam2base, T_link2optical=None):
        """
        Save calibration result to YAML.

        calibrateHandEye (eye-to-hand) returns T_{d435_color_optical_frame → base}.
        We need T_{d435_link → base} for publishing base→d435_link.

        Correction: T_{d435_link→base} = T_cam2base @ T_{d435_link→d435_color_optical}
        where T_link2optical = lookup_transform('d435_color_optical_frame', 'd435_link').
        """
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base

        if T_link2optical is not None:
            # Apply frame correction: T_cam2base is T_{optical→base},
            # T_link2optical transforms points from d435_link to optical.
            # T_{d435_link→base} = T_{optical→base} @ T_{d435_link→optical}
            T_save = T_cam2base @ T_link2optical
            self.get_logger().info('Applied d435_link→optical frame correction')
        else:
            T_save = T_cam2base
            self.get_logger().warn(
                'Saving raw optical→base result as d435_link→base (TF correction unavailable)')

        q = Rotation.from_matrix(T_save[:3, :3]).as_quat()

        result = {
            'frame_id': 'base',
            'child_frame_id': 'd435_link',
            'translation': {
                'x': float(T_save[0, 3]),
                'y': float(T_save[1, 3]),
                'z': float(T_save[2, 3]),
            },
            'rotation': {
                'x': float(q[0]),
                'y': float(q[1]),
                'z': float(q[2]),
                'w': float(q[3]),
            },
            '_cam2base_R': R_cam2base.tolist(),
            '_cam2base_t': t_cam2base.tolist(),
        }

        with open(self.output_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False)

        self.get_logger().info(f'Result saved to {self.output_path}')
        self.get_logger().info(
            f'd435_link position in base: '
            f'x={T_save[0,3]:.4f}, y={T_save[1,3]:.4f}, z={T_save[2,3]:.4f}')

    def run(self):
        """Main calibration loop."""
        self.get_logger().info('Waiting for camera info...')
        deadline = time.time() + 10.0
        while self.camera_matrix is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.camera_matrix is None:
            self.get_logger().error('Camera info not received. Is RealSense running?')
            return

        self.get_logger().info('Waiting for TCP pose...')
        deadline = time.time() + 5.0
        while self.latest_tcp_pose is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_tcp_pose is None:
            self.get_logger().error('TCP pose not received. Is tcp_pose_broadcaster running?')
            return

        samples = []
        for i, joint_angles in enumerate(self.poses):
            self.get_logger().info(
                f'Moving to pose {i+1}/{len(self.poses)}: {[f"{a:.3f}" for a in joint_angles]}')

            if not self._move_to_joints(joint_angles):
                self.get_logger().warn(f'Pose {i+1} failed, skipping')
                continue

            # Wait for robot to settle (with live preview)
            pose_label = f'Pose {i+1}/{len(self.poses)} (samples: {len(samples)})'
            self._settle_and_preview(1.5, label=f'{pose_label} - settling...')

            # Get TCP pose
            with self.pose_lock:
                tcp_pose = self.latest_tcp_pose
            if tcp_pose is None:
                self.get_logger().warn(f'No TCP pose at pose {i+1}, skipping')
                continue
            T_base_eef = self._pose_to_matrix(tcp_pose)

            # Detect AprilTag
            T_cam_tag = self._detect_tag(num_frames=10, label=pose_label)
            if T_cam_tag is None:
                self.get_logger().warn(
                    f'Tag not detected at pose {i+1}, skipping. '
                    'Ensure tag is visible from camera.')
                continue

            samples.append((T_base_eef, T_cam_tag))
            self.get_logger().info(
                f'  Captured sample {len(samples)}: '
                f'tag at {T_cam_tag[:3,3]} in camera frame')

        self.get_logger().info(f'Collected {len(samples)} samples')

        if len(samples) < 5:
            self.get_logger().error(
                f'Need at least 5 samples, got {len(samples)}. '
                'Check that tag is visible and poses are reachable.')
            return

        self.get_logger().info('Looking up d435_link→d435_color_optical_frame TF...')
        T_link2optical = self._lookup_link_to_optical()

        self.get_logger().info('Solving calibration...')
        R, t = self._solve_calibration(samples)
        self._save_result(R, t, T_link2optical)
        self.get_logger().info('Calibration complete! Run publish_tf.launch.py to publish TF.')


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCollector()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
