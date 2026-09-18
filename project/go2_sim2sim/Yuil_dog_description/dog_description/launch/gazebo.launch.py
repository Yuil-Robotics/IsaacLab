from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory('dog_description'))
    ros_gz_share = Path(get_package_share_directory('ros_gz_sim'))
    urdf_path = package_share / 'urdf' / 'TOTAL ASSY_4차_URDF_sample.urdf'
    robot_description = urdf_path.read_text(encoding='utf-8')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(ros_gz_share / 'launch' / 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    spawn_model = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'empty',
            '-file', str(urdf_path),
            '-name', 'dog',
            '-allow_renaming', 'true',
        ],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        TimerAction(period=3.0, actions=[spawn_model]),
    ])
