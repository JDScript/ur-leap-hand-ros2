#!/usr/bin/env python3
"""
Hand-eye calibration collector and solver.
Eye-to-hand: camera fixed, AprilTag on EEF.
Moves robot through predefined joint poses via MoveIt, detects AprilTag,
solves for base->camera transform using cv2.calibrateHandEye.

This version:
- Runs ROS spinning in a background daemon thread so the OpenCV preview
  stays live during MoveIt motion.
- Captures additional samples during motion (synchronized via TF buffer
  lookup at image timestamp), filtered by pose-delta thresholds.
- Surfaces calibration diagnostics: per-method consistency error in mm,
  per-axis std of T_eef_tag, sample-pose coverage, loud WARN/ERROR.
"""

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
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration
from ament_index_python.packages import get_package_share_directory

import tf2_ros

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from cv_bridge import CvBridge


def _rotation_angle_deg(R_a, R_b):
    """Angular distance between two rotation matrices, in degrees."""
    R_rel = R_a.T @ R_b
    cos = (np.trace(R_rel) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


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

        # Motion + in-motion sampling params
        self.velocity_scaling = float(cfg.get('velocity_scaling', 0.15))
        self.acceleration_scaling = float(cfg.get('acceleration_scaling', 0.15))
        self.min_translation_delta = float(cfg.get('min_translation_delta', 0.02))
        self.min_rotation_delta_deg = float(cfg.get('min_rotation_delta_deg', 5.0))
        self.max_total_samples = int(cfg.get('max_total_samples', 100))
        self.gripper_tf_frame = cfg.get('gripper_tf_frame', 'tool0_controller')
        self.enable_in_motion_sampling = bool(cfg.get('enable_in_motion_sampling', True))
        self.anchor_frames = int(cfg.get('anchor_frames', 10))

        # Image source
        self.image_topic = cfg.get('image_topic', '/camera/cam0/color/image_raw')
        self.camera_info_topic = cfg.get('camera_info_topic', '/camera/cam0/color/camera_info')
        self.use_rectified_image = bool(cfg.get('use_rectified_image', False))

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

        # Live preview state
        self._preview_label = ''
        self._preview_lock = threading.Lock()

        # TF buffer (cache 30s of history so image-timestamp lookups work in motion)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveIt action client
        self.moveit_client = ActionClient(self, MoveGroup, '/move_action')

        # Subscribers
        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic,
                                 self._camera_info_cb, 5)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose',
                                 self._tcp_cb, 10)

        # Background spinner
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spinner_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name='ros-spinner')

        self.get_logger().info('HandEyeCollector initialized')
        self.get_logger().info(f'Loaded {len(self.poses)} calibration poses')
        self.get_logger().info(
            f'Sampling: in_motion={self.enable_in_motion_sampling}, '
            f'anchor_frames={self.anchor_frames}, '
            f'vel={self.velocity_scaling}, '
            f'min_dt={self.min_translation_delta*100:.1f}cm, '
            f'min_dR={self.min_rotation_delta_deg:.1f}deg, '
            f'max_samples={self.max_total_samples}, gripper_tf={self.gripper_tf_frame}')

    # ---- Lifecycle ----------------------------------------------------------

    def start_spin(self):
        self._spinner_thread.start()

    def stop_spin(self):
        try:
            self._executor.shutdown()
        except Exception:
            pass

    # ---- Subscribers --------------------------------------------------------

    def _image_cb(self, msg):
        with self.image_lock:
            self.latest_image = msg

    def _camera_info_cb(self, msg):
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
            if self.use_rectified_image:
                # Image is already rectified by RealSense — distortion must
                # be treated as zero, otherwise estimatePoseSingleMarkers will
                # over-correct and give biased results.
                D = np.zeros_like(D) if D.size > 0 else np.zeros(5)
            self.camera_matrix = K
            self.dist_coeffs = D
            self.get_logger().info(
                f'Camera intrinsics from {self.camera_info_topic}:\n'
                f'  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}\n'
                f'  D={D.tolist()}  (use_rectified_image={self.use_rectified_image})')

    def _tcp_cb(self, msg):
        with self.pose_lock:
            self.latest_tcp_pose = msg.pose

    # ---- Geometry helpers ---------------------------------------------------

    def _pose_to_matrix(self, pose):
        """geometry_msgs/Pose -> 4x4 numpy matrix"""
        T = np.eye(4)
        q = [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def _tf_to_matrix(self, transform):
        """geometry_msgs/Transform -> 4x4 numpy matrix"""
        T = np.eye(4)
        t = transform.translation
        r = transform.rotation
        T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    # ---- MoveIt -------------------------------------------------------------

    def _move_to_joints(self, joint_angles, in_motion_cb=None):
        """
        Send joint space goal to MoveIt and wait for result, polling
        non-blockingly so the live preview keeps rendering.

        in_motion_cb: optional callable(); invoked once per polling tick
        (~10ms) while motion is in progress. Use this to capture in-motion
        samples and refresh preview.

        Returns True on success.
        """
        if not self.moveit_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveIt action server not available')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = self.get_parameter('move_group').value
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
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

        send_future = self.moveit_client.send_goal_async(goal)
        # Poll until accepted (preview still ticks) — short timeout for accept.
        accept_deadline = time.time() + 15.0
        while not send_future.done() and time.time() < accept_deadline:
            if in_motion_cb is not None:
                in_motion_cb()
            else:
                self._render_latest_preview()
            time.sleep(0.01)
        if not send_future.done():
            self.get_logger().error('Goal send timed out')
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        # Poll for execution result.
        result_future = goal_handle.get_result_async()
        motion_deadline = time.time() + 60.0
        while not result_future.done() and time.time() < motion_deadline:
            if in_motion_cb is not None:
                in_motion_cb()
            else:
                self._render_latest_preview()
            time.sleep(0.01)

        if not result_future.done():
            self.get_logger().error('Motion timed out')
            return False

        error_code = result_future.result().result.error_code.val
        if error_code == 1:
            return True
        else:
            self.get_logger().warn(f'MoveIt error code: {error_code}')
            return False

    # ---- Detection / preview ------------------------------------------------

    def _detect_in_image(self, img):
        """Run AprilTag detection on a BGR image. Returns (corners, ids)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._cv2_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._aruco_params)
        return corners, ids

    def _marker_object_points(self):
        """4 marker corners in marker frame, ordered to match ArUco corner order
        (TL, TR, BR, BL) — see OpenCV's _getSingleMarkerObjectPoints."""
        s = self.marker_size / 2.0
        return np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

    def _estimate_first_marker(self, corners):
        """
        Estimate (rvec, tvec, reproj_err_px) for the first marker.

        Uses solvePnPGeneric with SOLVEPNP_IPPE_SQUARE. IPPE returns BOTH
        ambiguous solutions for a planar square; we pick the one with
        smaller reprojection error. This avoids the silent pose-flip that
        the deprecated estimatePoseSingleMarkers can produce — a likely
        cause of cm-scale absolute bias even when consistency std is small.

        Returns None on failure.
        """
        try:
            obj_pts = self._marker_object_points()
            img_pts = corners[0].reshape(-1, 2).astype(np.float32)
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n_sol == 0:
                return None
            errs_arr = np.array(errs).flatten()
            best = int(np.argmin(errs_arr))
            rvec = np.asarray(rvecs[best]).reshape(3)
            tvec = np.asarray(tvecs[best]).reshape(3)
            return rvec, tvec, float(errs_arr[best])
        except Exception:
            return None

    def _draw_overlay(self, img, corners, ids, rvec=None, tvec=None, label=''):
        vis = img.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if rvec is not None and tvec is not None:
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
        return vis

    def _set_preview_label(self, label):
        with self._preview_lock:
            self._preview_label = label

    def _show_placeholder(self, label, message):
        """Show a gray frame with status text — used when no image available."""
        vis = np.full((360, 640, 3), 40, dtype=np.uint8)
        cv2.putText(vis, message, (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(vis, f'image topic: {self.image_topic}', (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis, f'camera_info: {self.camera_info_topic}', (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if label:
            cv2.putText(vis, label, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow('AprilTag Calibration', vis)
        cv2.waitKey(1)

    def _render_latest_preview(self, label=None):
        """
        Render one frame of the live preview using the most recent image.
        Safe to call frequently (e.g., from a polling loop). cv2 GUI calls
        always happen on the main thread (caller is the main thread).

        If no image / camera_info has arrived yet, render a placeholder so
        the OpenCV window always pops up — silent early return is unhelpful
        when debugging "preview never appears" issues.
        """
        if label is not None:
            self._set_preview_label(label)
        with self._preview_lock:
            current_label = self._preview_label

        with self.image_lock:
            img_msg = self.latest_image

        if img_msg is None:
            # Periodic warn so the terminal also surfaces the problem
            now = time.time()
            if now - getattr(self, '_last_no_image_warn', 0.0) > 5.0:
                self.get_logger().warn(
                    f'No image yet on {self.image_topic} — is the topic published?')
                self._last_no_image_warn = now
            self._show_placeholder(current_label,
                                   f'Waiting for image on {self.image_topic}')
            return

        if self.camera_matrix is None:
            self._show_placeholder(current_label,
                                   f'Waiting for camera_info on {self.camera_info_topic}')
            return

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self._show_placeholder(current_label, f'cv_bridge error: {e}')
            return

        corners, ids = self._detect_in_image(img)
        rvec = tvec = None
        if ids is not None and len(ids) > 0:
            est = self._estimate_first_marker(corners)
            if est is not None:
                rvec, tvec, _ = est
        vis = self._draw_overlay(img, corners, ids, rvec, tvec, label=current_label)
        cv2.imshow('AprilTag Calibration', vis)
        cv2.waitKey(1)

    def _settle_and_preview(self, duration, label=''):
        """Wait for robot to settle; preview ticks the whole time."""
        self._set_preview_label(label)
        end = time.time() + duration
        while time.time() < end:
            self._render_latest_preview()
            time.sleep(0.02)

    def _detect_tag_averaged(self, num_frames=10, label=''):
        """
        Capture num_frames distinct images, detect tag in each, average
        the result (the robot is stationary).
        Returns (T_cam_tag (4x4), mean_reproj_err_px) or (None, None).
        """
        if self.camera_matrix is None:
            self.get_logger().error('Camera intrinsics not yet received')
            return None, None

        rvecs_list = []
        tvecs_list = []
        reproj_errs = []
        deadline = time.time() + 5.0
        last_img_stamp = None

        while len(rvecs_list) < num_frames and time.time() < deadline:
            with self.image_lock:
                img_msg = self.latest_image
            if img_msg is None:
                time.sleep(0.02)
                continue
            img_stamp = (img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)
            if img_stamp == last_img_stamp:
                time.sleep(0.01)
                continue
            last_img_stamp = img_stamp

            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            corners, ids = self._detect_in_image(img)

            # Always render the live overlay so the user sees what's happening
            running_label = f'{label} | averaging {len(rvecs_list)}/{num_frames}'
            self._set_preview_label(running_label)
            self._render_latest_preview()

            if ids is None:
                continue
            if self.marker_id >= 0:
                mask = (ids.flatten() == self.marker_id)
                if not np.any(mask):
                    continue
                corners = [corners[i] for i in range(len(ids)) if mask[i]]
            est = self._estimate_first_marker(corners)
            if est is None:
                continue
            rvec, tvec, err_px = est
            rvecs_list.append(rvec)
            tvecs_list.append(tvec)
            reproj_errs.append(err_px)

        if len(rvecs_list) < 3:
            self.get_logger().warn(f'Only {len(rvecs_list)} frames detected, need at least 3')
            return None, None

        t_avg = np.mean(tvecs_list, axis=0)
        quats = []
        for rvec in rvecs_list:
            R, _ = cv2.Rodrigues(rvec)
            quats.append(Rotation.from_matrix(R).as_quat())
        R_avg = Rotation.from_quat(np.array(quats)).mean().as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_avg
        T[:3, 3] = t_avg
        return T, float(np.mean(reproj_errs))

    # ---- In-motion sampling -------------------------------------------------

    def _try_capture_in_motion(self, samples, last_accepted, status_cb=None):
        """
        Process one image (if a new one is available) and possibly accept
        an in-motion sample. Always renders the preview so the user sees
        live feedback during motion.

        - samples: list to append (T_base_eef, T_cam_tag) to.
        - last_accepted: dict with 't' (np.ndarray shape (3,)) and 'R' (3x3)
          keys, or empty. Updated in place when a sample is accepted.
        - status_cb: optional callable(label_str) used to refresh preview label.
        """
        # Always render the live preview — this is the main reason this
        # callback ticks even if no new sample is accepted.
        self._render_latest_preview()

        if self.camera_matrix is None:
            return
        if len(samples) >= self.max_total_samples:
            return

        with self.image_lock:
            img_msg = self.latest_image
        if img_msg is None:
            return

        # Track stamp to avoid double-processing the same image
        img_stamp = (img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)
        if getattr(self, '_last_motion_stamp', None) == img_stamp:
            return
        self._last_motion_stamp = img_stamp

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception:
            return
        corners, ids = self._detect_in_image(img)
        if ids is None or len(ids) == 0:
            return
        if self.marker_id >= 0:
            mask = (ids.flatten() == self.marker_id)
            if not np.any(mask):
                return
            corners = [corners[i] for i in range(len(ids)) if mask[i]]
        est = self._estimate_first_marker(corners)
        if est is None:
            return
        rvec, tvec, reproj_err_px = est
        # Track reproj error per accepted sample (high → intrinsics/marker
        # mismatch). Stored on instance so the solver summary can report it.
        self._last_reproj_err = reproj_err_px

        # Synchronize pose to image: query TF at image timestamp
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'base', self.gripper_tf_frame, img_msg.header.stamp,
                timeout=Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return
        T_base_eef = self._tf_to_matrix(tf_stamped.transform)

        R_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_tag
        T_cam_tag[:3, 3] = tvec

        # Pose-delta filter
        t_eef = T_base_eef[:3, 3]
        R_eef = T_base_eef[:3, :3]
        if last_accepted:
            dt = np.linalg.norm(t_eef - last_accepted['t'])
            dR = _rotation_angle_deg(last_accepted['R'], R_eef)
            if dt < self.min_translation_delta and dR < self.min_rotation_delta_deg:
                return

        samples.append((T_base_eef, T_cam_tag))
        last_accepted['t'] = t_eef
        last_accepted['R'] = R_eef

        if status_cb is not None:
            status_cb(len(samples))

    # ---- TF lookup for cam0_link → optical ---------------------------------

    def _lookup_link_to_optical(self):
        """
        Look up T_{cam0_link → cam0_color_optical_frame} from the TF tree.
        Returns a 4x4 numpy matrix (T_optical_link), or None if unavailable.
        """
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    'cam0_color_optical_frame', 'cam0_link', Time())
                T = self._tf_to_matrix(tf_stamped.transform)
                t = tf_stamped.transform.translation
                self.get_logger().info(
                    f'Got cam0_link→optical TF: t=[{t.x:.4f},{t.y:.4f},{t.z:.4f}]')
                return T
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                time.sleep(0.1)
        self.get_logger().warn(
            'Could not look up cam0_link→cam0_color_optical_frame. '
            'Is the RealSense driver running? Saving raw calibration without TF correction.')
        return None

    # ---- Solve --------------------------------------------------------------

    def _solve_calibration(self, samples):
        """
        Solve eye-to-hand calibration from samples list of (T_base_eef, T_cam_tag).
        For eye-to-hand: pass inverted FK poses to calibrateHandEye.
        Returns dict with keys: method, R, t, consistency_error_mm,
        T_eef_tag_mean_mm (np.ndarray shape (3,)), num_samples.
        """
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for T_base_eef, T_cam_tag in samples:
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
        # DANIILIDIS isn't always available in older OpenCV builds; add safely.
        if hasattr(cv2, 'CALIB_HAND_EYE_DANIILIDIS'):
            methods['DANIILIDIS'] = cv2.CALIB_HAND_EYE_DANIILIDIS

        best_method = None
        best_error = float('inf')
        best_R = None
        best_t = None
        best_T_eef_tag_mean = None

        self.get_logger().info('=== Calibration solver diagnostics ===')

        for name, method in methods.items():
            try:
                R, t = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method)

                # Consistency check: T_eef_tag = inv(T_base_eef) @ T_cam2base @ T_cam_tag
                # should be constant across all samples (tag is rigidly attached
                # to the EEF chain).
                T_cam2base = np.eye(4)
                T_cam2base[:3, :3] = R
                T_cam2base[:3, 3] = t.flatten()

                t_eef_tag_list = []
                for T_b_e, T_c_t in samples:
                    T_eef_tag = np.linalg.inv(T_b_e) @ T_cam2base @ T_c_t
                    t_eef_tag_list.append(T_eef_tag[:3, 3])
                t_arr = np.array(t_eef_tag_list)
                axis_std_mm = np.std(t_arr, axis=0) * 1000.0
                axis_mean_mm = np.mean(t_arr, axis=0) * 1000.0
                error = float(axis_std_mm.mean())

                self.get_logger().info(
                    f'  [{name:7s}] consistency std (mm) per axis: '
                    f'x={axis_std_mm[0]:.2f} y={axis_std_mm[1]:.2f} z={axis_std_mm[2]:.2f} '
                    f'| mean(mm) x={axis_mean_mm[0]:.1f} y={axis_mean_mm[1]:.1f} z={axis_mean_mm[2]:.1f}')

                if error < best_error:
                    best_error = error
                    best_method = name
                    best_R = R
                    best_t = t
                    best_T_eef_tag_mean = axis_mean_mm
            except Exception as e:
                self.get_logger().warn(f'  [{name:7s}] failed: {e}')

        self.get_logger().info(
            f'=== Best method: {best_method} (mean axis std = {best_error:.2f} mm) ===')
        self.get_logger().info(
            f'    Best T_eef_tag mean (mm): '
            f'x={best_T_eef_tag_mean[0]:.1f} y={best_T_eef_tag_mean[1]:.1f} z={best_T_eef_tag_mean[2]:.1f}  '
            f'(this is the inferred fixed offset from EEF to AprilTag)')

        if best_error > 20.0:
            self.get_logger().error(
                f'CONSISTENCY ERROR HIGH ({best_error:.1f} mm). '
                'Data is not self-consistent. Likely causes: '
                '(1) FK / URDF chain has errors at the EEF; '
                '(2) tcp_pose_broadcaster reports a different frame than expected; '
                '(3) tag detection is noisy (poor lighting / blur). '
                'Inspect per-axis std above to spot which axis is worst.')
        elif best_error > 5.0:
            self.get_logger().warn(
                f'CONSISTENCY ERROR ELEVATED ({best_error:.2f} mm). '
                'Data is mostly consistent but expect a few mm of error. '
                'If end result looks far off, check intrinsics and marker_size.')
        else:
            self.get_logger().info(
                f'Consistency error within ~5 mm. If end result still looks wrong, the '
                'issue is systematic bias (intrinsics, marker_size) not data noise.')

        # Reprojection-error stats (per-sample px error from PnP fit).
        # Independent of hand-eye math — measures whether intrinsics +
        # marker_size + corner detection agree at the pixel level. >2px
        # mean strongly suggests intrinsics or marker_size mismatch.
        reproj_errs = [e for e in getattr(self, '_sample_reproj_errs', [])
                       if e is not None and not np.isnan(e)]
        if reproj_errs:
            reproj_arr = np.array(reproj_errs)
            reproj_mean = float(reproj_arr.mean())
            reproj_max = float(reproj_arr.max())
            self.get_logger().info(
                f'Reprojection error across {len(reproj_arr)} samples: '
                f'mean={reproj_mean:.2f}px, max={reproj_max:.2f}px')
            if reproj_mean > 2.0:
                self.get_logger().warn(
                    f'REPROJECTION ERROR HIGH ({reproj_mean:.2f}px). '
                    'PnP can\'t fit the marker corners well. Most likely: '
                    'intrinsics (fx/fy/cx/cy) wrong, marker_size wrong, or '
                    'image distortion not handled correctly. Compare K from '
                    'log line "Camera intrinsics from ..." against an external '
                    'calibration (chessboard).')
        else:
            reproj_mean = float('nan')
            reproj_max = float('nan')

        return {
            'method': best_method,
            'R': best_R,
            't': best_t.flatten(),
            'consistency_error_mm': best_error,
            'T_eef_tag_mean_mm': best_T_eef_tag_mean,
            'num_samples': len(samples),
            'reproj_err_mean_px': reproj_mean,
            'reproj_err_max_px': reproj_max,
        }

    # ---- Sample diagnostics -------------------------------------------------

    def _log_sample_coverage(self, samples):
        """Print spatial/angular spread of T_base_eef across samples."""
        if len(samples) < 2:
            return
        ts = np.array([s[0][:3, 3] for s in samples])
        bbox = ts.max(axis=0) - ts.min(axis=0)

        # Angular spread: compare each rotation to the first, take max.
        R0 = samples[0][0][:3, :3]
        max_angle = 0.0
        for T_b_e, _ in samples:
            ang = _rotation_angle_deg(R0, T_b_e[:3, :3])
            if ang > max_angle:
                max_angle = ang

        tag_dists = np.array([np.linalg.norm(s[1][:3, 3]) for s in samples])
        self.get_logger().info(
            f'Sample coverage: N={len(samples)}, '
            f'EEF bbox(cm) x={bbox[0]*100:.1f} y={bbox[1]*100:.1f} z={bbox[2]*100:.1f}, '
            f'max EEF rotation {max_angle:.1f}°, '
            f'tag-distance min/mean/max(cm) '
            f'{tag_dists.min()*100:.1f}/{tag_dists.mean()*100:.1f}/{tag_dists.max()*100:.1f}')

        if bbox.max() < 0.05:
            self.get_logger().warn(
                'EEF translation spread < 5 cm — calibration may be poorly conditioned.')
        if max_angle < 30.0:
            self.get_logger().warn(
                f'EEF orientation spread < 30° (got {max_angle:.1f}°) — '
                'add more diverse rotations.')

    # ---- Save ---------------------------------------------------------------

    def _xform_to_dict(self, T):
        """4x4 matrix → {translation: {x,y,z}, rotation: {x,y,z,w}} dict."""
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        return {
            'translation': {
                'x': float(T[0, 3]),
                'y': float(T[1, 3]),
                'z': float(T[2, 3]),
            },
            'rotation': {
                'x': float(q[0]),
                'y': float(q[1]),
                'z': float(q[2]),
                'w': float(q[3]),
            },
        }

    def _save_result(self, result, T_link2optical=None):
        """
        Save calibration result to YAML.

        Top-level fields (frame_id, child_frame_id, translation, rotation)
        are kept exactly as before for tf_publisher.py compatibility.

        Additional `diagnostics` block stores method, error, sample count,
        and the supporting transforms (base→optical raw, eef→tag inferred,
        link→optical from URDF) so the run is auditable from the YAML alone.
        """
        R_cam2base = result['R']
        t_cam2base = result['t']

        T_base_optical = np.eye(4)
        T_base_optical[:3, :3] = R_cam2base
        T_base_optical[:3, 3] = t_cam2base

        if T_link2optical is not None:
            # T_base_link = T_base_optical @ T_optical_link
            T_base_link = T_base_optical @ T_link2optical
            self.get_logger().info('Applied cam0_link→optical frame correction')
        else:
            T_base_link = T_base_optical
            self.get_logger().warn(
                'Saving raw optical→base result as cam0_link→base (TF correction unavailable)')

        # Inferred fixed offset from EEF to AprilTag (in tool0 frame, mm)
        eef_tag_mm = result['T_eef_tag_mean_mm']

        out = {
            # ---- tf_publisher.py reads these (do not rename) ----
            'frame_id': 'base',
            'child_frame_id': 'cam0_link',
            **self._xform_to_dict(T_base_link),

            # ---- Auditable diagnostics ----
            'diagnostics': {
                'method': result['method'],
                'consistency_error_mm': float(result['consistency_error_mm']),
                'reproj_err_mean_px': float(result.get('reproj_err_mean_px', float('nan'))),
                'reproj_err_max_px': float(result.get('reproj_err_max_px', float('nan'))),
                'num_samples': int(result['num_samples']),
                'image_topic': self.image_topic,
                'use_rectified_image': bool(self.use_rectified_image),
                'gripper_tf_frame': self.gripper_tf_frame,

                # Raw calibration output: pose of cam0_color_optical_frame in base
                'base_to_color_optical': self._xform_to_dict(T_base_optical),

                # Inferred fixed AprilTag offset from EEF (tool0 frame), in meters
                'eef_to_tag': {
                    'translation': {
                        'x': float(eef_tag_mm[0] / 1000.0),
                        'y': float(eef_tag_mm[1] / 1000.0),
                        'z': float(eef_tag_mm[2] / 1000.0),
                    },
                    'distance_mm': float(np.linalg.norm(eef_tag_mm)),
                },

                # URDF transform that was composed in: T_optical_link
                'cam0_link_to_optical': (
                    self._xform_to_dict(T_link2optical)
                    if T_link2optical is not None else None),
            },
        }

        with open(self.output_path, 'w') as f:
            yaml.dump(out, f, default_flow_style=False, sort_keys=False)

        self.get_logger().info(f'Result saved to {self.output_path}')
        self._print_summary(result, T_base_optical, T_base_link, T_link2optical)

    def _print_summary(self, result, T_base_optical, T_base_link, T_link2optical):
        """Prominent end-of-run summary block."""
        eef_tag_mm = result['T_eef_tag_mean_mm']
        eef_tag_norm = float(np.linalg.norm(eef_tag_mm))

        def fmt_xform(T, label):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            return (
                f'  {label}\n'
                f'    translation [m]:  x={T[0,3]:+.4f}  y={T[1,3]:+.4f}  z={T[2,3]:+.4f}\n'
                f'    quat [xyzw]:      {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}')

        reproj_mean = result.get('reproj_err_mean_px', float('nan'))
        reproj_max = result.get('reproj_err_max_px', float('nan'))
        block = [
            '',
            '============================================================',
            '  CALIBRATION RESULT SUMMARY',
            '============================================================',
            f'  Method:           {result["method"]}',
            f'  Consistency std:  {result["consistency_error_mm"]:.2f} mm  (per-axis mean)',
            f'  Reproj error:     mean={reproj_mean:.2f}px  max={reproj_max:.2f}px',
            f'  Samples used:     {result["num_samples"]}',
            f'  Image topic:      {self.image_topic}  (rectified={self.use_rectified_image})',
            '',
            fmt_xform(T_base_optical, 'T_base_color_optical  (raw calibration output)'),
            '',
            fmt_xform(T_base_link, 'T_base_cam0_link      (saved → tf_publisher)'),
            '',
            '  T_tool0_tag           (inferred fixed AprilTag offset on EEF)',
            f'    translation [mm]: x={eef_tag_mm[0]:+.1f}  y={eef_tag_mm[1]:+.1f}  z={eef_tag_mm[2]:+.1f}',
            f'    magnitude:        {eef_tag_norm:.1f} mm  ({eef_tag_norm/10:.2f} cm)',
            '    >> Sanity check: measure your physical tag-to-tool0 distance.',
            '    >> If it disagrees by >1cm, FK chain or marker_size is suspect.',
        ]
        if T_link2optical is not None:
            block += [
                '',
                fmt_xform(T_link2optical,
                          'T_optical_cam0_link   (URDF, used in correction)'),
            ]
        block += [
            '============================================================',
            '',
        ]
        self.get_logger().info('\n'.join(block))

    # ---- Frame-source sanity check -----------------------------------------

    def _diagnose_tool0_vs_tool0_controller(self):
        """
        Compare the two candidate gripper frames. If they disagree, calibration
        results depend on which one we use, and a downstream comparison against
        URDF (which uses tool0) will show a fixed offset equal to the diff.
        """
        try:
            tf_a = self.tf_buffer.lookup_transform(
                'base', 'tool0', Time(), timeout=Duration(seconds=1.0))
            tf_b = self.tf_buffer.lookup_transform(
                'base', 'tool0_controller', Time(), timeout=Duration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Could not compare tool0 vs tool0_controller: {e}')
            return

        Ta = self._tf_to_matrix(tf_a.transform)
        Tb = self._tf_to_matrix(tf_b.transform)
        dt_mm = (Ta[:3, 3] - Tb[:3, 3]) * 1000.0
        dR_deg = _rotation_angle_deg(Ta[:3, :3], Tb[:3, :3])
        mag_mm = float(np.linalg.norm(dt_mm))

        self.get_logger().info(
            f'GRIPPER FRAME CHECK   base→tool0 vs base→tool0_controller:\n'
            f'  translation diff (mm): x={dt_mm[0]:+.2f}  y={dt_mm[1]:+.2f}  z={dt_mm[2]:+.2f}  '
            f'(magnitude {mag_mm:.2f} mm)\n'
            f'  rotation diff: {dR_deg:.3f} deg\n'
            f'  Currently using: {self.gripper_tf_frame}')
        if mag_mm > 5.0 or dR_deg > 0.5:
            self.get_logger().warn(
                f'tool0 and tool0_controller DIFFER by {mag_mm:.1f}mm / {dR_deg:.2f}deg. '
                'Likely cause: a TCP offset is set on the UR teach pendant, OR the UR '
                'controller and URDF use different DH parameters. Calibration with '
                f'gripper_tf_frame={self.gripper_tf_frame} will be biased relative to '
                'whatever frame downstream consumers (e.g., URDF mesh) use.')

    # ---- Main loop ----------------------------------------------------------

    def run(self):
        self.get_logger().info('Waiting for camera info...')
        deadline = time.time() + 10.0
        while self.camera_matrix is None and time.time() < deadline:
            self._render_latest_preview('Waiting for camera info...')
            time.sleep(0.05)
        if self.camera_matrix is None:
            self.get_logger().error('Camera info not received. Is RealSense running?')
            return

        self.get_logger().info('Waiting for TCP pose...')
        deadline = time.time() + 5.0
        while self.latest_tcp_pose is None and time.time() < deadline:
            self._render_latest_preview('Waiting for TCP pose...')
            time.sleep(0.05)
        if self.latest_tcp_pose is None:
            self.get_logger().error('TCP pose not received. Is tcp_pose_broadcaster running?')
            return

        self._diagnose_tool0_vs_tool0_controller()

        samples = []
        last_accepted = {}
        self._last_motion_stamp = None
        self._sample_reproj_errs = []

        def status_cb(n):
            self._set_preview_label(f'In-motion samples: {n}/{self.max_total_samples}')
            err = getattr(self, '_last_reproj_err', float('nan'))
            self._sample_reproj_errs.append(err)
            self.get_logger().info(
                f'  [in-motion] accepted sample {n}  (reproj_err={err:.2f}px)')

        for i, joint_angles in enumerate(self.poses):
            self.get_logger().info(
                f'Moving to pose {i+1}/{len(self.poses)}: '
                f'{[f"{a:.3f}" for a in joint_angles]}  (samples so far: {len(samples)})')
            self._set_preview_label(f'Moving to pose {i+1}/{len(self.poses)}')

            in_motion_cb = (
                (lambda: self._try_capture_in_motion(
                    samples, last_accepted, status_cb=status_cb))
                if self.enable_in_motion_sampling else None
            )
            if not self._move_to_joints(joint_angles, in_motion_cb=in_motion_cb):
                self.get_logger().warn(f'Pose {i+1} failed, skipping')
                continue

            # Settle + averaged anchor sample
            anchor_label = f'Pose {i+1}/{len(self.poses)} anchor (samples: {len(samples)})'
            self._settle_and_preview(1.0, label=f'{anchor_label} - settling')

            # Use TF lookup at "now" for consistency with in-motion samples,
            # rather than the broadcaster topic.
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    'base', self.gripper_tf_frame, Time(),
                    timeout=Duration(seconds=0.5))
                T_base_eef = self._tf_to_matrix(tf_stamped.transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(
                    f'TF lookup base→{self.gripper_tf_frame} failed at pose {i+1}: {e}')
                continue

            T_cam_tag, anchor_reproj_err = self._detect_tag_averaged(
                num_frames=self.anchor_frames, label=anchor_label)
            if T_cam_tag is None:
                self.get_logger().warn(
                    f'Tag not detected at anchor {i+1}, skipping. '
                    'Ensure tag is visible from camera.')
                continue

            samples.append((T_base_eef, T_cam_tag))
            self._sample_reproj_errs.append(anchor_reproj_err)
            last_accepted['t'] = T_base_eef[:3, 3]
            last_accepted['R'] = T_base_eef[:3, :3]
            tag_dist_cm = np.linalg.norm(T_cam_tag[:3, 3]) * 100
            self.get_logger().info(
                f'  [anchor] sample {len(samples)}: tag distance = {tag_dist_cm:.1f} cm  '
                f'(mean reproj_err over {self.anchor_frames} frames = {anchor_reproj_err:.2f}px)')

            if len(samples) >= self.max_total_samples:
                self.get_logger().info(
                    f'Reached max_total_samples ({self.max_total_samples}), stopping.')
                break

        self.get_logger().info(f'Collected {len(samples)} samples total')

        if len(samples) < 5:
            self.get_logger().error(
                f'Need at least 5 samples, got {len(samples)}. '
                'Check that tag is visible and poses are reachable.')
            return

        self._log_sample_coverage(samples)

        self.get_logger().info('Looking up cam0_link→cam0_color_optical_frame TF...')
        T_link2optical = self._lookup_link_to_optical()

        self.get_logger().info('Solving calibration...')
        result = self._solve_calibration(samples)
        self._save_result(result, T_link2optical)
        self.get_logger().info('Calibration complete! Run publish_tf.launch.py to publish TF.')


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCollector()
    node.start_spin()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.stop_spin()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
