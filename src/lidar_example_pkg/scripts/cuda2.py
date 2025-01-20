import rospy
import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker
import open3d as o3d
import pandas as pd


class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('lidar_process_node', anonymous=False)
        self.subscriber = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback)
        self.cloud_publisher = rospy.Publisher('/processed_points', PointCloud2, queue_size=10)
        self.marker_publisher = rospy.Publisher('/processed_bboxes', MarkerArray, queue_size=10)

    def callback(self, msg):
        try:
            rospy.loginfo("Received PointCloud2 data")
            point_cloud = self.convert_ros_to_open3d(msg)
            processed_cloud, bbox_objects = self.process_pointcloud(point_cloud)
            ros_cloud = self.convert_open3d_to_ros(processed_cloud, msg.header)
            marker_array = self.create_marker_array(bbox_objects)
            self.cloud_publisher.publish(ros_cloud)
            self.marker_publisher.publish(marker_array)
            rospy.loginfo("Published processed data")
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")

    def process_pointcloud(self, pcd):
        # 다운샘플링
        pcd = pcd.voxel_down_sample(voxel_size=0.5)

        # 평면 분할 (RANSAC)
        inliers = self.ransac_gpu(pcd, distance_threshold=0.3, num_iterations=500)
        # NumPy 배열을 정수 리스트로 변환
        inliers = inliers.tolist()  # NumPy 배열 → Python 리스트
        pcd = pcd.select_by_index(inliers, invert=True)

        # DBSCAN 클러스터링
        labels = np.array(pcd.cluster_dbscan(eps=0.60, min_points=30, print_progress=False))

        # 바운딩 박스 생성
        indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()
        bbox_objects = []
        for i in range(len(indexes)):
            nb_points = len(pcd.select_by_index(indexes[i]).points)
            if 50 <= nb_points <= 300:
                sub_cloud = pcd.select_by_index(indexes[i])
                bbox_object = sub_cloud.get_axis_aligned_bounding_box()
                bbox_objects.append(bbox_object)

        return pcd, bbox_objects


    def ransac_gpu(self, pcd, distance_threshold, num_iterations):
        points = torch.tensor(np.asarray(pcd.points), device='cuda')
        X_Y = points[:, :2]
        Z = points[:, 2]

        best_inliers = []
        for _ in range(num_iterations):
            indices = torch.randperm(X_Y.shape[0])[:3]
            selected_points = X_Y[indices]
            selected_z = Z[indices]

            # 평면 모델 추정
            A = torch.cat((selected_points, torch.ones((3, 1), device='cuda')), dim=1)
            result = torch.linalg.lstsq(A, selected_z.view(-1, 1))
            plane_coeffs = result.solution

            # 거리 계산
            distances = torch.abs((X_Y @ plane_coeffs[:2]) + plane_coeffs[2] - Z)
            inliers = torch.nonzero(distances < distance_threshold).squeeze()

            # 최적의 inliers 업데이트
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        return best_inliers.cpu().numpy()

    def convert_ros_to_open3d(self, ros_cloud):
        points = []
        for p in point_cloud2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        return cloud

    def convert_open3d_to_ros(self, open3d_cloud, header):
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


if __name__ == '__main__':
    processor = PointCloudProcessor()
    processor.spin()
