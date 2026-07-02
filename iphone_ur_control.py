#!/usr/bin/env python3
"""
iPhone to UR Robot Control Bridge (Standalone)
Converts iPhone Record3D pose data to MoveIt Servo twist commands
Handles controller switching and record3d launching
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit_msgs.srv import ServoCommandType
import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess
import signal
import sys
import time
import atexit


class iPhoneURControlBridge(Node):
    def __init__(self):
        super().__init__("iphone_ur_control_bridge")

        # Declare parameters
        self.declare_parameter("input_pose_topic", "/record3d/device_0/camera_pose")
        self.declare_parameter("output_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("linear_scale", 2.0)
        self.declare_parameter("angular_scale", 1.0)
        self.declare_parameter("dead_zone_linear", 0.01)  # m/s
        self.declare_parameter("dead_zone_angular", 0.02)  # rad/s
        self.declare_parameter("max_linear_velocity", 0.2)  # m/s
        self.declare_parameter("max_angular_velocity", 0.5)  # rad/s
        self.declare_parameter("control_frame", "base_link")
        self.declare_parameter("smoothing_factor", 0.2)  # EMA smoothing for velocity
        self.declare_parameter(
            "pose_smoothing_factor", 0.3
        )  # EMA smoothing for pose before differentiation
        self.declare_parameter("max_linear_acceleration", 1.0)  # m/s^2
        self.declare_parameter("max_angular_acceleration", 2.0)  # rad/s^2
        self.declare_parameter(
            "pose_timeout", 0.5
        )  # seconds - send zero velocity if no pose updates

        # Get parameters
        self.input_pose_topic = self.get_parameter("input_pose_topic").value
        self.output_twist_topic = self.get_parameter("output_twist_topic").value
        self.linear_scale = self.get_parameter("linear_scale").value
        self.angular_scale = self.get_parameter("angular_scale").value
        self.dead_zone_linear = self.get_parameter("dead_zone_linear").value
        self.dead_zone_angular = self.get_parameter("dead_zone_angular").value
        self.max_linear_vel = self.get_parameter("max_linear_velocity").value
        self.max_angular_vel = self.get_parameter("max_angular_velocity").value
        self.control_frame = self.get_parameter("control_frame").value
        self.smoothing_factor = self.get_parameter("smoothing_factor").value
        self.pose_smoothing_factor = self.get_parameter("pose_smoothing_factor").value
        self.max_linear_accel = self.get_parameter("max_linear_acceleration").value
        self.max_angular_accel = self.get_parameter("max_angular_acceleration").value
        self.pose_timeout = self.get_parameter("pose_timeout").value

        # State variables for velocity computation
        self.last_pose = None
        self.last_pose_time = None
        self.first_message = True

        # Smoothed pose (for pre-filtering before differentiation)
        self.smoothed_position = None
        self.smoothed_rotation = None

        # Smoothed velocity (exponential moving average)
        self.smoothed_linear_vel = np.zeros(3)
        self.smoothed_angular_vel = np.zeros(3)

        # Previous velocity for acceleration limiting
        self.prev_linear_vel = np.zeros(3)
        self.prev_angular_vel = np.zeros(3)

        # Current command to publish
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.control_frame

        # Subscribe to iPhone camera pose
        self.pose_sub = self.create_subscription(
            PoseStamped, self.input_pose_topic, self.pose_callback, 10
        )

        # Publish twist commands
        self.twist_pub = self.create_publisher(
            TwistStamped, self.output_twist_topic, 10
        )

        # Create timer to publish commands at high frequency (100Hz)
        # This maintains smooth control even if iPhone pose is only 60Hz
        self.publish_timer = self.create_timer(0.01, self.publish_command)

        # Create service client for switching servo command type
        self.switch_command_type_client = self.create_client(
            ServoCommandType, "/servo_node/switch_command_type"
        )

        # Wait for servo service
        self.get_logger().info("Waiting for servo service...")
        if self.switch_command_type_client.wait_for_service(timeout_sec=5.0):
            self.set_command_type_to_twist()
        else:
            self.get_logger().warn(
                "Servo service not available. Make sure servo node is running."
            )

        self.get_logger().info("=" * 60)
        self.get_logger().info("iPhone-UR Control Bridge started (Velocity Mode)")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Input: {self.input_pose_topic}")
        self.get_logger().info(f"  Output: {self.output_twist_topic}")
        self.get_logger().info(
            f"  Scaling: linear={self.linear_scale}, angular={self.angular_scale}"
        )
        self.get_logger().info(
            f"  Dead zones: linear={self.dead_zone_linear}m/s, angular={self.dead_zone_angular}rad/s"
        )
        self.get_logger().info(
            f"  Max velocities: linear={self.max_linear_vel}m/s, angular={self.max_angular_vel}rad/s"
        )
        self.get_logger().info(f"  Smoothing factor: {self.smoothing_factor}")
        self.get_logger().info(f"  Control frame: {self.control_frame}")
        self.get_logger().info(f"  Pose smoothing: {self.pose_smoothing_factor}")
        self.get_logger().info(
            f"  Max acceleration: linear={self.max_linear_accel}m/s², angular={self.max_angular_accel}rad/s²"
        )
        self.get_logger().info(f"  Pose timeout: {self.pose_timeout}s")
        self.get_logger().info(f"  Command publish rate: 100 Hz")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Waiting for iPhone pose data...")
        self.get_logger().info("VELOCITY MODE: Robot follows iPhone movement speed")

    def set_command_type_to_twist(self):
        """Set servo command type to TWIST"""
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST

        self.get_logger().info("Setting servo command type to TWIST...")
        future = self.switch_command_type_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info("✓ Servo command type set to TWIST")
        else:
            self.get_logger().error("✗ Failed to set servo command type")

    def pose_callback(self, msg: PoseStamped):
        """Handle incoming iPhone pose messages and compute velocity"""
        current_time = self.get_clock().now()

        # Extract current pose
        curr_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ]
        )
        curr_quat = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        curr_rotation = R.from_quat(curr_quat)

        # Skip first message, initialize smoothed pose
        if self.first_message:
            self.smoothed_position = curr_position.copy()
            self.smoothed_rotation = curr_rotation
            self.last_pose_time = current_time
            self.first_message = False
            self.get_logger().info("✓ First pose received. Ready for control!")
            return

        # Compute time delta
        dt = (current_time - self.last_pose_time).nanoseconds / 1e9

        if dt < 0.001:  # Avoid division by very small numbers
            return

        # === POSE PRE-FILTERING ===
        # Apply EMA smoothing to position
        alpha_pose = self.pose_smoothing_factor
        smoothed_position_new = (
            alpha_pose * curr_position + (1 - alpha_pose) * self.smoothed_position
        )

        # Apply SLERP smoothing to orientation (quaternion interpolation)
        # SLERP(q1, q2, t) interpolates between q1 and q2
        # For EMA-like effect: new_smooth = SLERP(old_smooth, current, alpha)
        try:
            smoothed_rotation_new = R.from_quat(
                self.slerp(
                    self.smoothed_rotation.as_quat(),
                    curr_rotation.as_quat(),
                    alpha_pose,
                )
            )
        except Exception:
            # Fallback to direct assignment if SLERP fails
            smoothed_rotation_new = curr_rotation

        # === VELOCITY COMPUTATION FROM SMOOTHED POSE ===
        # Compute linear velocity from smoothed position
        linear_vel = (smoothed_position_new - self.smoothed_position) / dt

        # Compute angular velocity from smoothed orientation
        delta_rot = self.smoothed_rotation.inv() * smoothed_rotation_new
        rotvec = delta_rot.as_rotvec()
        angular_vel = rotvec / dt

        # Update smoothed pose for next iteration
        self.smoothed_position = smoothed_position_new
        self.smoothed_rotation = smoothed_rotation_new

        # Apply dead zone filtering
        linear_vel, angular_vel = self.apply_dead_zone(linear_vel, angular_vel)

        # === ACCELERATION LIMITING (ANTI-JERK) ===
        linear_vel = self.limit_acceleration(
            linear_vel, self.prev_linear_vel, dt, self.max_linear_accel
        )
        angular_vel = self.limit_acceleration(
            angular_vel, self.prev_angular_vel, dt, self.max_angular_accel
        )

        # Store for next iteration
        self.prev_linear_vel = linear_vel.copy()
        self.prev_angular_vel = angular_vel.copy()

        # === VELOCITY SMOOTHING (secondary EMA on velocity) ===
        alpha = self.smoothing_factor
        self.smoothed_linear_vel = (
            alpha * linear_vel + (1 - alpha) * self.smoothed_linear_vel
        )
        self.smoothed_angular_vel = (
            alpha * angular_vel + (1 - alpha) * self.smoothed_angular_vel
        )

        # Scale SMOOTHED velocities
        scaled_linear_vel = self.smoothed_linear_vel * self.linear_scale
        scaled_angular_vel = self.smoothed_angular_vel * self.angular_scale

        # Clamp to maximum velocities
        linear_magnitude = np.linalg.norm(scaled_linear_vel)
        if linear_magnitude > self.max_linear_vel:
            scaled_linear_vel = (
                scaled_linear_vel / linear_magnitude * self.max_linear_vel
            )

        angular_magnitude = np.linalg.norm(scaled_angular_vel)
        if angular_magnitude > self.max_angular_vel:
            scaled_angular_vel = (
                scaled_angular_vel / angular_magnitude * self.max_angular_vel
            )

        # Update current twist command (will be published by timer at 100Hz)
        self.current_twist.header.stamp = current_time.to_msg()
        self.current_twist.twist.linear.x = scaled_linear_vel[0]
        self.current_twist.twist.linear.y = scaled_linear_vel[1]
        self.current_twist.twist.linear.z = scaled_linear_vel[2]
        self.current_twist.twist.angular.x = scaled_angular_vel[0]
        self.current_twist.twist.angular.y = scaled_angular_vel[1]
        self.current_twist.twist.angular.z = scaled_angular_vel[2]

        # Update last pose time
        self.last_pose_time = current_time

    def publish_command(self):
        """Publish twist command at high frequency (100Hz) with timeout detection"""
        # Check for pose timeout
        if self.last_pose_time is not None:
            time_since_last_pose = (
                self.get_clock().now() - self.last_pose_time
            ).nanoseconds / 1e9
            if time_since_last_pose > self.pose_timeout:
                # Pose data is stale, send zero velocity
                zero_twist = TwistStamped()
                zero_twist.header.stamp = self.get_clock().now().to_msg()
                zero_twist.header.frame_id = self.control_frame
                self.twist_pub.publish(zero_twist)
                return

        # Update timestamp and publish current command
        self.current_twist.header.stamp = self.get_clock().now().to_msg()
        self.twist_pub.publish(self.current_twist)

    def slerp(self, quat1, quat2, t):
        """Spherical linear interpolation between two quaternions"""
        # Ensure quaternions are numpy arrays
        q1 = np.array(quat1)
        q2 = np.array(quat2)

        # Compute dot product
        dot = np.dot(q1, q2)

        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Clamp dot product
        dot = np.clip(dot, -1.0, 1.0)

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        # Calculate angle between quaternions
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        # Compute SLERP
        q2_orth = q2 - q1 * dot
        q2_orth = q2_orth / np.linalg.norm(q2_orth)

        return q1 * np.cos(theta) + q2_orth * np.sin(theta)

    def limit_acceleration(self, velocity, prev_velocity, dt, max_accel):
        """Limit acceleration to prevent sudden jerks"""
        if dt < 0.001:
            return velocity

        # Compute acceleration
        accel = (velocity - prev_velocity) / dt
        accel_magnitude = np.linalg.norm(accel)

        # If acceleration exceeds limit, clamp it
        if accel_magnitude > max_accel:
            # Limit acceleration magnitude
            accel = accel / accel_magnitude * max_accel
            # Compute new velocity based on limited acceleration
            velocity = prev_velocity + accel * dt

        return velocity

    def apply_dead_zone(self, linear_vel, angular_vel):
        """Apply dead zone filtering to prevent jitter"""
        # Linear dead zone
        linear_magnitude = np.linalg.norm(linear_vel)
        if linear_magnitude < self.dead_zone_linear:
            linear_vel = np.zeros(3)

        # Angular dead zone
        angular_magnitude = np.linalg.norm(angular_vel)
        if angular_magnitude < self.dead_zone_angular:
            angular_vel = np.zeros(3)

        return linear_vel, angular_vel


# Global variables for cleanup
record3d_process = None
node = None


def switch_controller(target_controller, stop_controller):
    """Switch ROS 2 controllers"""
    print(f"\n🔄 Switching controller to {target_controller}...")
    cmd = [
        "ros2",
        "control",
        "switch_controllers",
        "--activate",
        target_controller,
        "--deactivate",
        stop_controller,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Controller switched to {target_controller}")
            return True
        else:
            print(f"✗ Failed to switch controller: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error switching controller: {e}")
        return False


def cleanup():
    """Cleanup function called on exit"""
    global record3d_process, node

    print("\n" + "=" * 60)
    print("Shutting down iPhone-UR Control...")
    print("=" * 60)

    # Send zero velocities BEFORE shutting down ROS
    if node is not None and rclpy.ok():
        try:
            print("Sending zero velocities...")
            twist_msg = TwistStamped()
            twist_msg.header.stamp = node.get_clock().now().to_msg()
            twist_msg.header.frame_id = "base_link"
            for _ in range(10):
                node.twist_pub.publish(twist_msg)
                time.sleep(0.01)
            print("✓ Zero velocities sent")
        except Exception as e:
            print(f"⚠ Could not send zero velocities: {e}")

    # Shutdown ROS
    if rclpy.ok():
        rclpy.shutdown()

    # Stop record3d process
    if record3d_process is not None:
        print("Stopping record3d node...")
        record3d_process.terminate()
        try:
            record3d_process.wait(timeout=5)
            print("✓ Record3d node stopped")
        except subprocess.TimeoutExpired:
            print("Force killing record3d node...")
            record3d_process.kill()

    # Switch back to trajectory controller
    switch_controller(
        "scaled_joint_trajectory_controller", "forward_velocity_controller"
    )

    print("=" * 60)
    print("✓ Cleanup complete")
    print("=" * 60)


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\n\nReceived interrupt signal...")
    cleanup()
    sys.exit(0)


def main(args=None):
    global record3d_process, node

    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("iPhone-UR Robot Control - Standalone")
    print("=" * 60)

    # Switch to velocity controller
    if not switch_controller(
        "forward_velocity_controller", "scaled_joint_trajectory_controller"
    ):
        print("⚠ Warning: Failed to switch to velocity controller")
        print("  Make sure ros2_control is running and controllers are loaded")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return

    # Launch record3d node
    print("\n📱 Starting record3d node...")
    try:
        record3d_process = subprocess.Popen(
            ["ros2", "run", "record3d", "record3d_node"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("✓ Record3d node started")
        time.sleep(2)  # Give it time to start
    except Exception as e:
        print(f"✗ Failed to start record3d: {e}")
        print("  Make sure record3d package is built and sourced")
        cleanup()
        return

    # Initialize ROS and start control bridge
    print("\n🤖 Starting control bridge...")
    rclpy.init(args=args)

    try:
        node = iPhoneURControlBridge()
        print("\n✓ All systems ready!")
        print("Move your iPhone to set reference position, then control the robot")
        print("Press Ctrl+C to stop\n")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
