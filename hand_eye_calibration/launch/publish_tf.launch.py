from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hand_eye_calibration',
            executable='tf_publisher.py',
            name='calibration_tf_publisher',
            output='screen',
        ),
    ])
