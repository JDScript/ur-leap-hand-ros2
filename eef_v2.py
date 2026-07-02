#!/usr/bin/env python3

import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    DisplayTrajectory,
)
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive


class MoveItClient(Node):
    def __init__(self):
        super().__init__("moveit_action_client")

        self.plan_client = ActionClient(self, MoveGroup, "/move_action")
        self.exec_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.display_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )

        self.planned_trajectory = None
        self.trajectory_start = None
        self._display_msg = None
        self._display_timer = None

    def _build_goal(self):
        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_arm"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.request.max_velocity_scaling_factor = 0.1
        goal.planning_options.plan_only = True

        pose = PoseStamped()
        pose.header.frame_id = "base"
        pose.pose.position.x = -0.5289
        pose.pose.position.y = -0.0587
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = -0.0767
        pose.pose.orientation.y = -0.03
        pose.pose.orientation.z = 0.9483
        pose.pose.orientation.w = -0.3065

        constraints = Constraints()

        pc = PositionConstraint()
        pc.header = pose.header
        pc.link_name = "tool0"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
        pc.constraint_region.primitives.append(box)
        pc.constraint_region.primitive_poses.append(pose.pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header = pose.header
        oc.link_name = "leap_hand_palm_lower"
        oc.orientation = pose.pose.orientation
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0

        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        goal.request.goal_constraints.append(constraints)

        return goal

    def _wait(self, future):
        """Block until an rclpy future completes (driven by the background spin thread)."""
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        event.wait()
        return future.result()

    def _stop_display(self):
        self._display_msg = None
        if self._display_timer is not None:
            self._display_timer.cancel()
            self._display_timer = None

    def _publish_display(self):
        if self._display_msg is not None:
            self.display_pub.publish(self._display_msg)

    def plan(self):
        self.get_logger().info("Waiting for move_action server...")
        self.plan_client.wait_for_server()

        goal = self._build_goal()
        self.get_logger().info("Planning...")

        future = self.plan_client.send_goal_async(goal)
        goal_handle = self._wait(future)
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait(result_future).result
        error_code = result.error_code.val

        if error_code != 1:
            self.get_logger().error(f"Planning failed (error_code={error_code})")
            return False

        self.planned_trajectory = result.planned_trajectory
        self.trajectory_start = result.trajectory_start
        n_points = len(
            self.planned_trajectory.joint_trajectory.points
        )
        self.get_logger().info(
            f"Planning succeeded — {n_points} waypoints, "
            f"duration {self.planned_trajectory.joint_trajectory.points[-1].time_from_start.sec}s"
        )

        display_msg = DisplayTrajectory()
        display_msg.trajectory_start = self.trajectory_start
        display_msg.trajectory.append(self.planned_trajectory)
        self._display_msg = display_msg

        # Compute trajectory duration so we republish only after the animation finishes
        last_point = self.planned_trajectory.joint_trajectory.points[-1]
        traj_duration = last_point.time_from_start.sec + last_point.time_from_start.nanosec * 1e-9
        period = max(traj_duration + 1.0, 3.0)  # at least 3s between publishes

        if self._display_timer is not None:
            self._display_timer.cancel()
        self._display_timer = self.create_timer(period, self._publish_display)
        # publish once immediately
        self.display_pub.publish(display_msg)

        self.get_logger().info(
            f"Trajectory visible in RViz (replaying every {period:.1f}s)"
        )

        return True

    def execute(self):
        if self.planned_trajectory is None:
            self.get_logger().error("No trajectory to execute")
            return False

        self.get_logger().info("Waiting for execute_trajectory server...")
        self.exec_client.wait_for_server()

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self.planned_trajectory

        self.get_logger().info("Executing...")
        future = self.exec_client.send_goal_async(goal)
        goal_handle = self._wait(future)
        if not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait(result_future).result
        error_code = result.error_code.val

        if error_code == 1:
            self.get_logger().info("Execution: SUCCESS")
            return True
        else:
            self.get_logger().error(f"Execution failed (error_code={error_code})")
            return False

    def run(self):
        # Spin in background so the display timer keeps firing during input()
        spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        spin_thread.start()

        while True:
            if not self.plan():
                choice = input("[r] re-plan  [q] quit: ").strip().lower()
                if choice == "r":
                    continue
                break

            choice = (
                input("[enter/e] execute  [r] re-plan  [q] quit: ").strip().lower()
            )
            if choice in ("", "e"):
                self._stop_display()
                self.execute()
                break
            elif choice == "r":
                self._stop_display()
                continue
            else:
                break


def main():
    rclpy.init()
    node = MoveItClient()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
