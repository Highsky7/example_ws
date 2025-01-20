import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import hdbscan
import pandas as pd


class PointCloudProcessor:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('lidar_process_node', anonymous=True)

        # 토픽 구독 및 발행 설정
        self.subscriber = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback)
        self.cloud_publisher = rospy.Publisher('/processed_points', PointCloud2, queue_size=10)
        self.marker_publisher = rospy.Publisher('/processed_bboxes', MarkerArray, queue_size=10)

    def callback(self, msg):
        """
        ROS 토픽 데이터 수신 시 호출되는 콜백 함수
        """
        rospy.loginfo("Received PointCloud2 data")

        # 1. PointCloud2 메시지를 Open3D 포인트 클라우드로 변환
        point_cloud = self.convert_ros_to_open3d(msg)

        # 2. Open3D를 이용한 포인트 클라우드 전처리
        processed_cloud, bbox_objects = self.process_pointcloud(point_cloud)

        # 3. 처리된 데이터를 다시 ROS PointCloud2 메시지로 변환
        ros_cloud = self.convert_open3d_to_ros(processed_cloud, msg.header)

        # 4. 바운딩 박스를 MarkerArray로 변환
        marker_array = self.create_marker_array(bbox_objects)

        # 5. 데이터 발행
        self.cloud_publisher.publish(ros_cloud)
        self.marker_publisher.publish(marker_array)

        rospy.loginfo("Published processed PointCloud2 data and bounding boxes")

    def process_pointcloud(self, pcd):
        """
        Open3D를 사용하여 포인트 클라우드 전처리 및 바운딩 박스 생성
        """
        # 다운샘플링
        pcd = pcd.voxel_down_sample(voxel_size=0.05)

        # 평면 분할 (RANSAC)
        inliers = self.ransac_segmentation(pcd, distance_threshold=0.3, num_iterations=500)
        pcd = pcd.select_by_index(inliers, invert=True)

        # 클러스터링 (HDBSCAN)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
        # clusterer.fit(np.array(pcd.points))
        # labels = clusterer.labels_
        
        # DBSCAN
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm: # for debug
            labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=30, print_progress=False)) # each Point gets its label. (noise points get -1 label)

        

        # 클러스터별 바운딩 박스 생성
        indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()
        bbox_objects = []
        for i in range(len(indexes)):
            nb_points = len(pcd.select_by_index(indexes[i]).points)
            if 50 <= nb_points <= 300:
                sub_cloud = pcd.select_by_index(indexes[i])
                bbox_object = sub_cloud.get_axis_aligned_bounding_box()
                bbox_objects.append(bbox_object)

        return pcd, bbox_objects

    def ransac_segmentation(self, pcd, distance_threshold, num_iterations):
        """
        RANSAC 알고리즘을 사용한 평면 분할
        """
        pcd_points = np.array(pcd.points)
        X_Y = pcd_points[:, :2]
        Z = pcd_points[:, 2]

        poly_features = PolynomialFeatures(degree=1, include_bias=True)
        X_poly = poly_features.fit_transform(X_Y)

        ransac = RANSACRegressor(min_samples=3, residual_threshold=distance_threshold, max_trials=num_iterations)
        ransac.fit(X_poly, Z)

        inlier_mask = ransac.inlier_mask_
        inliers = np.nonzero(inlier_mask)[0]

        rospy.loginfo(f"RANSAC found {len(inliers)} inliers")
        return inliers

    def convert_ros_to_open3d(self, ros_cloud):
        """
        ROS PointCloud2 메시지를 Open3D 포인트 클라우드로 변환
        """
        points = []
        for p in point_cloud2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True):
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

    def create_marker_array(self, bbox_objects):
        """
        바운딩 박스 데이터를 ROS MarkerArray 메시지로 변환
        """
        marker_array = MarkerArray()
        for i, bbox in enumerate(bbox_objects):
            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = bbox.get_center()[0]
            marker.pose.position.y = bbox.get_center()[1]
            marker.pose.position.z = bbox.get_center()[2]
            marker.scale.x = bbox.get_extent()[0]
            marker.scale.y = bbox.get_extent()[1]
            marker.scale.z = bbox.get_extent()[2]
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.id = i
            marker_array.markers.append(marker)
        return marker_array

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        processor = PointCloudProcessor()
        processor.spin()
    except rospy.ROSInterruptException:
        pass
