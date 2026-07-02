#!/usr/bin/env python3
"""
MoveIt Servo Control Script
Publishes continuous twist commands to control the robot via MoveIt Servo
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from moveit_msgs.srv import ServoCommandType
from moveit_msgs.msg import ServoStatus


class ServoTwistController(Node):
    def __init__(self):
        super().__init__("servo_twist_controller")

        # Create publisher for twist commands
        self.twist_pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10
        )

        # Create subscriber for servo status
        self.status_sub = self.create_subscription(
            ServoStatus, "/servo_node/status", self.status_callback, 10
        )

        # Track the latest servo status
        self.latest_status = None
        self.last_status_code = None

        # Create service clients
        self.switch_command_type_client = self.create_client(
            ServoCommandType, "/servo_node/switch_command_type"
        )

        # Wait for services
        self.get_logger().info("Waiting for servo services...")
        self.switch_command_type_client.wait_for_service(timeout_sec=5.0)

        # Set command type to TWIST
        self.set_command_type_to_twist()

        # Parameters for the twist command
        self.linear_x = -0.00  # m/s
        self.linear_y = 0.0
        self.linear_z = -0.03
        self.angular_x = 0.00
        self.angular_y = 0.00
        self.angular_z = 0.0

        # Duration to send commands (seconds)
        self.duration = 100.0

        # Publishing rate (Hz) - should match servo's publish_period (0.01s = 100Hz)
        self.publish_rate = 500.0

        # Create timer to publish commands
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_twist)

        # Track elapsed time
        self.start_time = self.get_clock().now()
        self.is_active = True

        self.get_logger().info(f"Publishing twist commands for {self.duration} seconds")
        self.get_logger().info(
            f"Linear velocity: ({self.linear_x}, {self.linear_y}, {self.linear_z}) m/s"
        )
        self.get_logger().info(
            f"Angular velocity: ({self.angular_x}, {self.angular_y}, {self.angular_z}) rad/s"
        )

    def status_callback(self, msg: ServoStatus):
        """Handle servo status updates"""
        self.latest_status = msg

        # Map status codes to readable names
        status_names = {
            0: "NO_WARNING",
            1: "DECELERATE_FOR_SINGULARITY",
            2: "HALT_FOR_SINGULARITY",
            3: "DECELERATE_FOR_COLLISION",
            4: "HALT_FOR_COLLISION",
            5: "JOINT_BOUND",
            6: "DECELERATE_FOR_LEAVING_SINGULARITY_REGION",
            -1: "INVALID",
        }

        status_name = status_names.get(msg.code, f"UNKNOWN({msg.code})")

        # Only print when status changes
        if msg.code != self.last_status_code:
            if msg.code == 0:
                self.get_logger().info(f"✓ Servo Status: {status_name}")
            elif msg.code in [3, 4]:  # Collision warnings
                self.get_logger().warn(f"⚠ Servo Status: {status_name} - {msg.message}")
            elif msg.code in [1, 2]:  # Singularity warnings
                self.get_logger().warn(f"⚠ Servo Status: {status_name} - {msg.message}")
            else:
                self.get_logger().warn(f"⚠ Servo Status: {status_name} - {msg.message}")

            self.last_status_code = msg.code

    def set_command_type_to_twist(self):
        """Set servo command type to TWIST (value 1)"""
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST  # Value: 1

        self.get_logger().info("Setting command type to TWIST...")
        future = self.switch_command_type_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info("✓ Command type set to TWIST successfully")
            else:
                self.get_logger().error("✗ Failed to set command type to TWIST")
        else:
            self.get_logger().warn("Service call timed out")

    def publish_twist(self):
        """Publish twist command at high frequency"""
        if not self.is_active:
            return

        # Check if duration has elapsed
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Print status every second
        if int(elapsed) != getattr(self, "_last_print_second", -1):
            self._last_print_second = int(elapsed)
            status_msg = "No status yet"
            if self.latest_status:
                status_names = {
                    0: "NO_WARNING",
                    1: "DECELERATE_FOR_SINGULARITY",
                    2: "HALT_FOR_SINGULARITY",
                    3: "DECELERATE_FOR_COLLISION",
                    4: "HALT_FOR_COLLISION",
                    5: "JOINT_BOUND",
                    6: "DECELERATE_FOR_LEAVING_SINGULARITY_REGION",
                    -1: "INVALID",
                }
                status_msg = status_names.get(
                    self.latest_status.code, f"UNKNOWN({self.latest_status.code})"
                )
            self.get_logger().info(
                f"[{int(elapsed)}s] Publishing commands... Current servo status: {status_msg}"
            )

        if elapsed >= self.duration:
            # Stop the robot by sending zero velocities
            self.get_logger().info("Duration elapsed, stopping robot...")
            self.send_zero_twist()
            self.is_active = False
            self.timer.cancel()

            # Give it a moment to send the stop command
            self.create_timer(0.5, self.shutdown_node)
            return

        # Create and publish twist message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"

        twist_msg.twist.linear.x = self.linear_x
        twist_msg.twist.linear.y = self.linear_y
        twist_msg.twist.linear.z = self.linear_z

        twist_msg.twist.angular.x = self.angular_x
        twist_msg.twist.angular.y = self.angular_y
        twist_msg.twist.angular.z = self.angular_z

        self.twist_pub.publish(twist_msg)

    def send_zero_twist(self):
        """Send zero velocities to stop the robot"""
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"

        # All velocities are zero by default

        # Publish multiple times to ensure it's received
        for _ in range(10):
            self.twist_pub.publish(twist_msg)

    def shutdown_node(self):
        """Shutdown the node gracefully"""
        self.get_logger().info("Shutting down...")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ServoTwistController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
