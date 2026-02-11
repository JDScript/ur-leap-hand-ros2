import os
import yaml

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def declare_arguments():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_rviz", default_value="true", description="Launch RViz?"
            ),
            DeclareLaunchArgument(
                "launch_servo", default_value="true", description="Launch Servo?"
            ),
            DeclareLaunchArgument(
                "load_obstacles",
                default_value="true",
                description="Load static obstacles?",
            ),
            DeclareLaunchArgument(
                "obstacles_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("custom_moveit_config"),
                        "config",
                        "obstacles.yaml",
                    ]
                ),
                description="Path to obstacles YAML file",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Using or not time from simulation",
            ),
            DeclareLaunchArgument(
                "publish_robot_description_semantic",
                default_value="true",
                description="MoveGroup publishes robot description semantic",
            ),
        ]
    )


def generate_launch_description():
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")
    load_obstacles = LaunchConfiguration("load_obstacles")
    obstacles_file = LaunchConfiguration("obstacles_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    publish_robot_description_semantic = LaunchConfiguration(
        "publish_robot_description_semantic"
    )

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="custom_moveit_config")
        .robot_description_semantic(Path("config") / "ur.srdf")
        .to_moveit_configs()
    )

    ld = LaunchDescription()
    ld.add_entity(declare_arguments())

    wait_robot_description = Node(
        package="ur_robot_driver",
        executable="wait_for_robot_description",
        output="screen",
    )
    ld.add_action(wait_robot_description)

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,
                "publish_robot_description_semantic": publish_robot_description_semantic,
            },
        ],
    )

    servo_yaml = load_yaml("custom_moveit_config", "config/ur_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}
    servo_node = Node(
        package="moveit_servo",
        condition=IfCondition(launch_servo),
        executable="servo_node",
        parameters=[
            moveit_config.to_dict(),
            servo_params,
        ],
        output="screen",
    )

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("custom_moveit_config"), "config", "moveit.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    obstacle_loader_node = Node(
        package="custom_moveit_config",
        executable="load_obstacles.py",
        name="obstacle_loader",
        output="screen",
        parameters=[
            {
                "obstacles_file": obstacles_file,
                "publish_rate": 1.0,
            }
        ],
        condition=IfCondition(load_obstacles),
    )

    ld.add_action(
        RegisterEventHandler(
            OnProcessExit(
                target_action=wait_robot_description,
                on_exit=[move_group_node, rviz_node, servo_node, obstacle_loader_node],
            )
        ),
    )

    return ld
