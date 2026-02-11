#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from rclpy.action.client import ClientGoalHandle
from rclpy.task import Future


class MoveItClient(Node):
    def __init__(self):
        super().__init__("moveit_action_client")

        self.client = ActionClient(self, MoveGroup, "/move_action")

    def send_goal(self):
        self.client.wait_for_server()

        goal = MoveGroup.Goal()
        goal.request.group_name = "ur_arm"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_acceleration_scaling_factor = 1.0
        goal.request.max_velocity_scaling_factor = 1.0
        goal.planning_options.replan_attempts = 5
        goal.planning_options.replan = True

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = 0.9
        pose.pose.position.y = 0.2
        pose.pose.position.z = 0.5
        pose.pose.orientation.w = 1.0

        constraints = Constraints()
        pc = PositionConstraint()
        pc.header = pose.header
        pc.link_name = "leap_hand_palm_lower"

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

        self.get_logger().info("Sending goal...")

        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future):
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        result_future: Future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future: Future):
        result = future.result().result
        error_code = result.error_code.val

        if error_code == 1:
            self.get_logger().info("Motion planning: SUCCESS")
        elif error_code == 99999:
            self.get_logger().error("Motion planning: FAILURE")
        elif error_code == -1:
            self.get_logger().error("Motion planning: PLANNING_FAILED")
        elif error_code == -2:
            self.get_logger().error("Motion planning: INVALID_MOTION_PLAN")
        elif error_code == -3:
            self.get_logger().error(
                "Motion planning: MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE"
            )
        elif error_code == -4:
            self.get_logger().error("Motion planning: CONTROL_FAILED")
        else:
            self.get_logger().error(f"Motion planning: UNKNOWN_ERROR_CODE {error_code}")

        rclpy.shutdown()


def main():
    rclpy.init()
    node = MoveItClient()
    node.send_goal()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
