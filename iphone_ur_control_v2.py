#!/usr/bin/env python3
"""
iPhone to UR Robot Control Bridge v2 (Pose Tracking)

Differences vs v1:
  - Sends target POSE to /servo_node/pose_target_cmds (servo POSE mode) instead
    of differentiating pose into a twist. No numerical differentiation = no
    noise amplification = no chatter / no servo whining.
  - Closed-loop in position: when the iPhone returns to its reference, the
    target pose returns to the robot's reference, so the arm comes back too.
  - Auto-clutch via stillness detection: when the iPhone has been stationary
    for `static_anchor_duration` seconds, the iPhone+robot reference is
    re-anchored to the current poses. Lets the user reposition without
    dragging the arm.

Safety:
  - `dry_run` defaults to True. In dry-run, target poses are published to
    ~/pose_target_debug only; the controller and servo command type are NOT
    switched. Use `ros2 topic echo /iphone_ur_pose_bridge/pose_target_debug`
    and Foxglove/RViz to confirm correctness before letting it drive the arm.
  - Robot EE pose is read via TF (`planning_frame -> ee_frame`), NOT from
    /tcp_pose_broadcaster/pose. The broadcaster publishes in `base` frame,
    not `base_link`, which differs by 180° yaw — silently sending those
    coordinates as `base_link` target was the v2-original runaway bug.
  - First anchor only happens after the iPhone has been static for
    `static_anchor_duration` seconds (prevents anchoring on ARKit init noise).
"""

import atexit
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R


class iPhoneURPoseBridge(Node):
    def __init__(self):
        super().__init__("iphone_ur_pose_bridge")

        self.declare_parameter("input_pose_topic", "/record3d/device_0/camera_pose")
        self.declare_parameter("output_pose_topic", "/servo_node/pose_target_cmds")
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("rotation_scale", 1.0)
        # Position uses heavier smoothing (noise rejection), rotation uses
        # lighter smoothing (responsiveness). Larger alpha = less smoothing.
        self.declare_parameter("pose_smoothing_factor", 0.2)
        self.declare_parameter("rotation_smoothing_factor", 0.5)
        self.declare_parameter("motion_deadband_pos", 0.008)   # 8mm; below this, no motion
        self.declare_parameter("motion_deadband_rot", 0.015)   # ~0.86°; below this, no rotation
        self.declare_parameter("static_position_threshold", 0.015)
        self.declare_parameter("static_rotation_threshold", 0.05)
        self.declare_parameter("static_anchor_duration", 1.0)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("max_step_position", 0.05)
        self.declare_parameter("dry_run", True)
        # When using the new pose-tracking servo config, scaled_joint_traj
        # ctrl is shared with MoveIt — no controller switching required.
        self.declare_parameter("manage_controllers", False)
        # auto_reanchor=False (default): only the FIRST anchor is automatic;
        #   subsequent re-anchors require pressing Enter in the terminal where
        #   this script is running.
        # auto_reanchor=True: re-anchor every time iPhone transitions
        #   moving -> still (acts like a soft auto-clutch).
        self.declare_parameter("auto_reanchor", False)

        self.input_pose_topic = self.get_parameter("input_pose_topic").value
        self.output_pose_topic = self.get_parameter("output_pose_topic").value
        self.planning_frame = self.get_parameter("planning_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.position_scale = self.get_parameter("position_scale").value
        self.rotation_scale = self.get_parameter("rotation_scale").value
        self.pose_smoothing_factor = self.get_parameter("pose_smoothing_factor").value
        self.rotation_smoothing_factor = self.get_parameter("rotation_smoothing_factor").value
        self.motion_deadband_pos = self.get_parameter("motion_deadband_pos").value
        self.motion_deadband_rot = self.get_parameter("motion_deadband_rot").value
        self.static_pos_thresh = self.get_parameter("static_position_threshold").value
        self.static_rot_thresh = self.get_parameter("static_rotation_threshold").value
        self.static_duration = self.get_parameter("static_anchor_duration").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.max_step_position = self.get_parameter("max_step_position").value
        self.dry_run = self.get_parameter("dry_run").value
        self.auto_reanchor = self.get_parameter("auto_reanchor").value

        # ---- iPhone pose state ----
        self.smoothed_iphone_pos = None
        self.smoothed_iphone_rot = None

        # ---- Robot EE pose state (from TF) ----
        self.latest_robot_pos = None
        self.latest_robot_rot = None

        # ---- Anchor state ----
        self.iphone_ref_pos = None
        self.iphone_ref_rot = None
        self.robot_ref_pos = None
        self.robot_ref_rot = None
        self.anchored = False
        self.anchor_time = None
        self.was_static = False

        # Manual re-anchor request flag (set by stdin thread, consumed by tick).
        self._manual_reanchor_lock = threading.Lock()
        self._manual_reanchor_pending = False

        # Sliding window (time, pos, quat) for stillness detection.
        self.motion_window = deque()

        self.last_target_pos = None
        self.last_target_rot = None

        # ---- TF for robot pose ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Subscriptions ----
        self.iphone_sub = self.create_subscription(
            PoseStamped, self.input_pose_topic, self.iphone_cb, 10
        )

        # ---- Publishers ----
        # Always publish to a debug topic so the user can `ros2 topic echo` /
        # visualize what would be sent.
        self.debug_pub = self.create_publisher(
            PoseStamped, "~/pose_target_debug", 10
        )
        if self.dry_run:
            self.real_pub = None
        else:
            self.real_pub = self.create_publisher(
                PoseStamped, self.output_pose_topic, 10
            )

        self.timer = self.create_timer(1.0 / self.publish_rate, self.tick)
        self.status_timer = self.create_timer(1.0, self.log_status)

        # ---- Servo command type switch (only in real mode) ----
        if not self.dry_run:
            self.switch_command_type_client = self.create_client(
                ServoCommandType, "/servo_node/switch_command_type"
            )
            self.get_logger().info("Waiting for servo service...")
            if self.switch_command_type_client.wait_for_service(timeout_sec=5.0):
                self.set_command_type_to_pose()
            else:
                self.get_logger().warn(
                    "Servo service not available. Targets will publish but "
                    "servo may not act on them."
                )

        mode_label = "DRY-RUN (debug topic only)" if self.dry_run else "LIVE (driving robot)"
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"iPhone-UR Pose Bridge v2 — {mode_label}")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  iPhone pose:   {self.input_pose_topic}")
        self.get_logger().info(f"  Robot pose:    TF {self.planning_frame} -> {self.ee_frame}")
        self.get_logger().info(f"  Real target:   {self.output_pose_topic} "
                                f"({'enabled' if not self.dry_run else 'DISABLED in dry_run'})")
        self.get_logger().info("  Debug target:  ~/pose_target_debug "
                               "(/iphone_ur_pose_bridge/pose_target_debug)")
        self.get_logger().info(
            f"  Scaling:       pos={self.position_scale}, rot={self.rotation_scale}"
        )
        self.get_logger().info(
            f"  Pose EMA:      pos_alpha={self.pose_smoothing_factor}, "
            f"rot_alpha={self.rotation_smoothing_factor}"
        )
        self.get_logger().info(
            f"  Static thresh: pos={self.static_pos_thresh}m, "
            f"rot={self.static_rot_thresh}rad, dur={self.static_duration}s"
        )
        self.get_logger().info(
            f"  Auto reanchor: {self.auto_reanchor} "
            f"({'auto-clutch on every still transition' if self.auto_reanchor else 'first anchor only; press Enter for manual reanchor'})"
        )
        self.get_logger().info("=" * 60)
        self.get_logger().info("Holding iPhone still until first anchor lock...")
        if not self.auto_reanchor:
            self.get_logger().info("Press <Enter> in this terminal to re-anchor at any time.")

        # Stdin listener: pressing Enter requests a manual re-anchor.
        self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._stdin_thread.start()

    def set_command_type_to_pose(self):
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.POSE
        self.get_logger().info("Setting servo command type to POSE...")
        future = self.switch_command_type_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None and future.result().success:
            self.get_logger().info("✓ Servo command type set to POSE")
        else:
            self.get_logger().error("✗ Failed to set servo command type to POSE")

    def iphone_cb(self, msg: PoseStamped):
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        rot = R.from_quat(quat)

        if self.smoothed_iphone_pos is None:
            self.smoothed_iphone_pos = pos
            self.smoothed_iphone_rot = rot
        else:
            self.smoothed_iphone_pos = (
                self.pose_smoothing_factor * pos
                + (1.0 - self.pose_smoothing_factor) * self.smoothed_iphone_pos
            )
            self.smoothed_iphone_rot = R.from_quat(
                slerp(
                    self.smoothed_iphone_rot.as_quat(),
                    rot.as_quat(),
                    self.rotation_smoothing_factor,
                )
            )

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.motion_window.append(
            (now_sec, self.smoothed_iphone_pos.copy(), self.smoothed_iphone_rot.as_quat())
        )
        cutoff = now_sec - self.static_duration
        while self.motion_window and self.motion_window[0][0] < cutoff:
            self.motion_window.popleft()

    def update_robot_pose_from_tf(self):
        """Look up the EE pose in the planning frame via TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.ee_frame,
                rclpy.time.Time(),  # latest
                timeout=Duration(seconds=0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return False

        self.latest_robot_pos = np.array(
            [t.transform.translation.x,
             t.transform.translation.y,
             t.transform.translation.z]
        )
        self.latest_robot_rot = R.from_quat(
            [t.transform.rotation.x,
             t.transform.rotation.y,
             t.transform.rotation.z,
             t.transform.rotation.w]
        )
        return True

    def is_iphone_static(self):
        """True iff every sample in the past `static_duration` window is within
        position+rotation thresholds of the most recent sample."""
        if len(self.motion_window) < 2:
            return False
        oldest_time = self.motion_window[0][0]
        newest_time = self.motion_window[-1][0]
        if newest_time - oldest_time < self.static_duration * 0.9:
            return False

        _, ref_pos, ref_quat = self.motion_window[-1]
        ref_rot = R.from_quat(ref_quat)
        for _, p, q in self.motion_window:
            if np.linalg.norm(p - ref_pos) > self.static_pos_thresh:
                return False
            if (ref_rot.inv() * R.from_quat(q)).magnitude() > self.static_rot_thresh:
                return False
        return True

    def _stdin_loop(self):
        while True:
            try:
                input()  # blocks until Enter
            except (EOFError, KeyboardInterrupt):
                return
            with self._manual_reanchor_lock:
                self._manual_reanchor_pending = True

    def reanchor(self):
        self.iphone_ref_pos = self.smoothed_iphone_pos.copy()
        self.iphone_ref_rot = self.smoothed_iphone_rot
        self.robot_ref_pos = self.latest_robot_pos.copy()
        self.robot_ref_rot = self.latest_robot_rot
        self.anchored = True
        self.anchor_time = self.get_clock().now()

    def tick(self):
        if self.smoothed_iphone_pos is None:
            return
        if not self.update_robot_pose_from_tf():
            return

        # Manual reanchor request from stdin thread.
        with self._manual_reanchor_lock:
            manual = self._manual_reanchor_pending
            self._manual_reanchor_pending = False

        if not self.anchored:
            # Wait for iPhone to be still (or a manual press) before first
            # anchor — avoids latching onto an ARKit-initialization-time pose.
            if not (manual or self.is_iphone_static()):
                return
            self.reanchor()
            self.was_static = True
            self.last_target_pos = self.robot_ref_pos.copy()
            self.last_target_rot = self.robot_ref_rot
            self.get_logger().info(
                f"✓ Initial anchor locked. EE @ ({self.robot_ref_pos[0]:+.3f}, "
                f"{self.robot_ref_pos[1]:+.3f}, {self.robot_ref_pos[2]:+.3f}) "
                f"in {self.planning_frame}"
            )
            self._publish_target(self.robot_ref_pos, self.robot_ref_rot)
            return

        if manual:
            self.reanchor()
            self.get_logger().info(
                f"⏎ Manual reanchor. EE @ ({self.robot_ref_pos[0]:+.3f}, "
                f"{self.robot_ref_pos[1]:+.3f}, {self.robot_ref_pos[2]:+.3f})"
            )

        is_static_now = self.is_iphone_static()

        if self.auto_reanchor and is_static_now and not self.was_static:
            # Edge: moving -> still. Auto-clutch.
            self.reanchor()
            self.get_logger().info(
                f"↺ Auto re-anchored. EE @ ({self.robot_ref_pos[0]:+.3f}, "
                f"{self.robot_ref_pos[1]:+.3f}, {self.robot_ref_pos[2]:+.3f})"
            )

        self.was_static = is_static_now

        # Tracking: target = robot_ref ⊕ (iphone_now ⊖ iphone_ref) in world frame.
        # Apply motion deadband: noise below threshold = no motion. Above
        # threshold, subtract deadband from magnitude so motion is continuous
        # at the boundary (no step jump when crossing it).
        delta_pos = self.smoothed_iphone_pos - self.iphone_ref_pos
        delta_pos_mag = float(np.linalg.norm(delta_pos))
        if delta_pos_mag < self.motion_deadband_pos:
            target_pos = self.robot_ref_pos.copy()
        else:
            effective_mag = (delta_pos_mag - self.motion_deadband_pos) * self.position_scale
            target_pos = self.robot_ref_pos + (delta_pos / delta_pos_mag) * effective_mag

        delta_rot = self.smoothed_iphone_rot * self.iphone_ref_rot.inv()
        delta_rot_mag = float(delta_rot.magnitude())
        if delta_rot_mag < self.motion_deadband_rot:
            target_rot = self.robot_ref_rot
        else:
            t_scale = (delta_rot_mag - self.motion_deadband_rot) / delta_rot_mag * self.rotation_scale
            effective_delta_quat = slerp(
                np.array([0.0, 0.0, 0.0, 1.0]),
                delta_rot.as_quat(),
                t_scale,
            )
            target_rot = R.from_quat(effective_delta_quat) * self.robot_ref_rot

        # Outlier guard: clamp single-tick step
        if self.last_target_pos is not None:
            step = target_pos - self.last_target_pos
            step_mag = np.linalg.norm(step)
            if step_mag > self.max_step_position:
                target_pos = self.last_target_pos + step / step_mag * self.max_step_position

        self.last_target_pos = target_pos.copy()
        self.last_target_rot = target_rot
        self._publish_target(target_pos, target_rot)

    def _publish_target(self, pos, rot):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.planning_frame
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        q = rot.as_quat()
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.debug_pub.publish(msg)
        if self.real_pub is not None:
            self.real_pub.publish(msg)

    def log_status(self):
        if self.smoothed_iphone_pos is None:
            self.get_logger().info("[status] waiting for iPhone pose...")
            return
        if self.latest_robot_pos is None:
            self.get_logger().info("[status] waiting for TF "
                                   f"{self.planning_frame}->{self.ee_frame}...")
            return
        if not self.anchored:
            window_span = 0.0
            if len(self.motion_window) >= 2:
                window_span = self.motion_window[-1][0] - self.motion_window[0][0]
            still = self.is_iphone_static()
            self.get_logger().info(
                f"[status] PRE-ANCHOR  static={still}  window={window_span:.2f}s"
            )
            return

        delta_pos = self.smoothed_iphone_pos - self.iphone_ref_pos
        delta_rot_mag = (self.smoothed_iphone_rot * self.iphone_ref_rot.inv()).magnitude()
        target_step = (
            np.linalg.norm(self.last_target_pos - self.robot_ref_pos)
            if self.last_target_pos is not None else 0.0
        )
        anchor_age = (self.get_clock().now() - self.anchor_time).nanoseconds * 1e-9

        # Raw iPhone noise (peak-to-peak) over the static-detection window —
        # use this to tune deadband / static thresholds.
        noise_pp = 0.0
        if len(self.motion_window) >= 2:
            positions = np.array([p for _, p, _ in self.motion_window])
            noise_pp = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))

        self.get_logger().info(
            f"[status] mode={'DRY' if self.dry_run else 'LIVE'} "
            f"iphone_dpos={np.linalg.norm(delta_pos)*1000:5.1f}mm "
            f"iphone_drot={np.degrees(delta_rot_mag):5.1f}deg "
            f"target_step={target_step*1000:5.1f}mm "
            f"noise_pp={noise_pp*1000:4.1f}mm "
            f"static={self.is_iphone_static()} "
            f"anchor_age={anchor_age:.1f}s"
        )


def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions [x,y,z,w]."""
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    b_orth = b - a * dot
    b_orth = b_orth / np.linalg.norm(b_orth)
    return a * np.cos(theta) + b_orth * np.sin(theta)


# Global handle for cleanup
node = None


def switch_controller(target_controller, stop_controller):
    print(f"\n🔄 Switching controller to {target_controller}...")
    cmd = [
        "ros2", "control", "switch_controllers",
        "--activate", target_controller,
        "--deactivate", stop_controller,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Controller switched to {target_controller}")
            return True
        print(f"✗ Failed to switch controller: {result.stderr}")
        return False
    except Exception as e:
        print(f"✗ Error switching controller: {e}")
        return False


def cleanup(dry_run, manage_controllers):
    print("\n" + "=" * 60)
    print("Shutting down iPhone-UR Pose Bridge...")
    print("=" * 60)
    if rclpy.ok():
        rclpy.shutdown()
    if not dry_run and manage_controllers:
        switch_controller(
            "scaled_joint_trajectory_controller", "forward_velocity_controller"
        )
    print("✓ Cleanup complete")


def main(args=None):
    global node

    rclpy.init(args=args)

    # Peek params before constructing the node so main() can decide what to do.
    pre = rclpy.create_node("_iphone_ur_pose_bridge_pre")
    pre.declare_parameter("dry_run", True)
    pre.declare_parameter("manage_controllers", False)
    dry_run = pre.get_parameter("dry_run").value
    manage_controllers = pre.get_parameter("manage_controllers").value
    pre.destroy_node()

    atexit.register(lambda: cleanup(dry_run, manage_controllers))
    signal.signal(signal.SIGINT,
                  lambda *_: (cleanup(dry_run, manage_controllers), sys.exit(0)))

    print("=" * 60)
    print(f"iPhone-UR Pose Bridge v2  (dry_run={dry_run}, "
          f"manage_controllers={manage_controllers})")
    print("=" * 60)

    if not dry_run and manage_controllers:
        if not switch_controller(
            "forward_velocity_controller", "scaled_joint_trajectory_controller"
        ):
            print("⚠ Failed to switch to velocity controller")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                return
        time.sleep(0.3)
    elif dry_run:
        print("DRY-RUN: not switching controllers, not switching servo mode.")
        print("         Targets will publish to ~/pose_target_debug only.")
        print("         Re-run with `--ros-args -p dry_run:=false` to drive the arm.")
    else:
        print("LIVE without controller management — assumes servo is running with")
        print("custom_ur_pose_tracking config (writes to scaled_joint_traj_ctrl).")

    try:
        node = iPhoneURPoseBridge()
        print("\n✓ Bridge running. Ctrl+C to stop.\n")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup(dry_run)


if __name__ == "__main__":
    main()
