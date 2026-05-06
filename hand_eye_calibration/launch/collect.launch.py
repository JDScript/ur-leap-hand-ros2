from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hand_eye_calibration',
            executable='collector.py',
            name='hand_eye_collector',
            output='screen',
            emulate_tty=True,
        ),
    ])
