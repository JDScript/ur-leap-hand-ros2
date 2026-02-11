from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("custom_ur_description")

    urdf_path = PathJoinSubstitution([pkg_share, "urdf", "ur_leap_hand.urdf.xacro"])

    rviz_config_path = PathJoinSubstitution([pkg_share, "rviz", "urdf.rviz"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("name", default_value="ur", description="Robot name"),
            DeclareLaunchArgument(
                "ur_type", default_value="ur10e", description="UR Robot Type"
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": Command(
                            [
                                "xacro ",
                                urdf_path,
                                " name:=",
                                LaunchConfiguration("name"),
                                " ur_type:=",
                                LaunchConfiguration("ur_type"),
                            ]
                        )
                    }
                ],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config_path],
                output="screen",
            ),
        ]
    )
