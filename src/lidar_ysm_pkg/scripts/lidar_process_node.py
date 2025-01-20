#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

class PointCloudProcessor:
    def __init__(self):
        #ROS node initialize
        rospy.init_node('lidar_process_node', anonymous=True)

        #topic subscribe, publish
        self.subscriber = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback)
        self.publisher = rospy.Publisher('/processed_points', PointCloud2, queue_size=10)

    
    def callback(self, msg):
        #log message for check
        rospy.loginfo("Received PointCloud2 data")
        '''
        data preprocess here
        '''
        
       # 1. PointCloud2 메시지를 Open3D 포인트 클라우드로 변환
        point_cloud = self.convert_ros_to_open3d(msg)

        # 2. Open3D를 이용한 전처리
        processed_cloud = self.open3d_point_cloud(point_cloud)

        # 3. 처리된 데이터를 다시 ROS PointCloud2 메시지로 변환
        ros_cloud = self.convert_open3d_to_ros(processed_cloud, msg.header)
        
        # 4. 전처리된 데이터를 발행
        #self.publisher.publish(msg)
        self.publisher.publish(ros_cloud)
        rospy.loginfo("Published processed PointCloud2 data")

    def open3d_point_cloud(self, cloud):
        """
        Open3D를 이용하여 포인트 클라우드 전처리 수행
        """
        # 다운샘플링 (Voxel Downsampling)
        voxel_size = 0.1  # 조정 가능
        cloud_downsampled = cloud.voxel_down_sample(voxel_size)

         # 노이즈 제거 (Statistical Outlier Removal)
        cl, ind = cloud_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cloud_denoised = cloud_downsampled.select_by_index(ind)

        # 평면 분할 (RANSAC Plane Segmentation)
        plane_model, inliers = cloud_denoised.segment_plane(distance_threshold=0.02,
                                                            ransac_n=3,
                                                            num_iterations=1000)
        cloud_segmented = cloud_denoised.select_by_index(inliers, invert=True)

        rospy.loginfo("Point cloud preprocessing completed")
        return cloud_segmented

    def convert_ros_to_open3d(self, ros_cloud):
        """
        ROS PointCloud2 메시지를 Open3D 포인트 클라우드로 변환
        """
        points = []
        for p in pc2.read_points(ros_cloud, skip_nans=True):
            points.append([p[0], p[1], p[2]])

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        return cloud
    
    def convert_open3d_to_ros(self, open3d_cloud, header):
        """
        Open3D 포인트 클라우드를 ROS PointCloud2 메시지로 변환
        """
        points = np.asarray(open3d_cloud.points)
        ros_cloud = PointCloud2()
        ros_cloud.header = header
        ros_cloud.height = 1
        ros_cloud.width = points.shape[0]
        ros_cloud.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        ros_cloud.is_bigendian = False
        ros_cloud.point_step = 12
        ros_cloud.row_step = ros_cloud.point_step * points.shape[0]
        ros_cloud.is_dense = True
        ros_cloud.data = np.asarray(points, np.float32).tobytes()
        return ros_cloud

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        processor.spin()
    except rospy.ROSInternalException:
        pass