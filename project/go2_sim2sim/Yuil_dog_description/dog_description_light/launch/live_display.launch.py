from pathlib import Path

from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory('dog_description'))
    package_prefix = Path(get_package_prefix('dog_description'))
    rsp_prefix = Path(get_package_prefix('robot_state_publisher'))

    urdf_path = package_share / 'urdf' / 'TOTAL ASSY_4차_URDF_sample.urdf'
    rviz_config = package_share / 'config' / 'display.rviz'
    reloader = package_prefix / 'lib' / 'dog_description' / 'urdf_live_reloader'
    robot_state_publisher = (
        rsp_prefix / 'lib' / 'robot_state_publisher' / 'robot_state_publisher'
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=[str(reloader), str(urdf_path), str(robot_state_publisher)],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            parameters=[{'rate': 30}],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', str(rviz_config), '-f', 'base'],
            output='screen',
        ),
    ])
