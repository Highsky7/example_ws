from setuptools import setup

setup(
    name='lidar_example_pkg',
    version='0.0.1',
    packages=['lidar_example_pkg'],
    install_requires=['rospy', 'sensor_msgs', 'visualization_msgs'],
    package_dir={'': 'scripts'},  # Python 스크립트 경로 설정
)
