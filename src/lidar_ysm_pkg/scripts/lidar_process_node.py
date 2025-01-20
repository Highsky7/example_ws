#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

# HDBSCAN 라이브러리
import hdbscan


class PointCloudProcessor:
    def __init__(self):
        # ROS node initialize
        rospy.init_node('lidar_process_node', anonymous=True)

        # Subscribe & Publish
        self.subscriber = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback)
        self.publisher = rospy.Publisher('/processed_points', PointCloud2, queue_size=10)

    def callback(self, msg):
        """
        LiDAR 토픽 '/velodyne_points'로부터 PointCloud2 메시지를 수신하면,
        아래의 전처리 파이프라인을 수행한 뒤 결과를 퍼블리시한다.
        """
        rospy.loginfo("Received PointCloud2 data")

        # 1. ROS -> Open3D 변환
        cloud_o3d = self.convert_ros_to_open3d(msg)

        # 2. 파이프라인 처리 (DownSample, Outlier 제거, Plane 분할, HDBSCAN, Bounding Box 추출 등)
        processed_cloud_o3d = self.process_point_cloud(cloud_o3d)

        # 3. 최종적으로 전처리된 Open3D 포인트 클라우드를 ROS PointCloud2로 변환
        ros_cloud = self.convert_open3d_to_ros(processed_cloud_o3d, msg.header)

        # 4. 퍼블리시
        self.publisher.publish(ros_cloud)
        rospy.loginfo("Published processed PointCloud2 data")

    def process_point_cloud(self, cloud_o3d):
        """
        Open3D 포인트 클라우드 객체에 대해
        1) 다운샘플링
        2) 통계적 이상점 제거
        3) RANSAC 평면 분할
        4) HDBSCAN 클러스터링
        5) 클러스터별 Bounding Box
        순으로 수행하고, 최종적으로 필요한 포인트만 남긴 Cloud를 반환한다.
        """
        # --------------------
        # 1. Voxel DownSampling
        # --------------------
        voxel_size = 0.1  # 상황에 맞춰 조정
        cloud_downsampled = cloud_o3d.voxel_down_sample(voxel_size)
        rospy.loginfo("Voxel DownSample finished. Points: {} -> {}".format(
            len(cloud_o3d.points), len(cloud_downsampled.points)))

        # --------------------
        # 2. Outlier 제거 (Statistical Outlier Removal)
        # --------------------
        nb_neighbors = 20
        std_ratio = 2.0
        cl, inliers = cloud_downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        cloud_denoised = cloud_downsampled.select_by_index(inliers)
        rospy.loginfo("Outlier removal finished. Points: {} -> {}".format(
            len(cloud_downsampled.points), len(cloud_denoised.points)))

        # --------------------
        # 3. RANSAC Plane Segmentation
        #    (예: 도로/바닥면 등을 제거하기 위함)
        # --------------------
        distance_threshold = 0.3
        ransac_n = 3
        num_iterations = 1000
        plane_model, inliers = cloud_denoised.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        # plane_model: (a, b, c, d) 형태의 평면 방정식
        # inliers: plane에 해당하는 점들의 인덱스

        # 평면에 속하는 점, 아닌 점을 분리
        inlier_cloud = cloud_denoised.select_by_index(inliers)
        outlier_cloud = cloud_denoised.select_by_index(inliers, invert=True)

        rospy.loginfo("Plane segmentation finished.")
        rospy.loginfo("  Plane points: {}, Non-plane points: {}".format(
            len(inlier_cloud.points), len(outlier_cloud.points)))

        # 예: 바닥면(plane)은 제거하고, 나머지 포인트(outlier_cloud)에 대해서만 후속 처리
        # 필요한 경우 inlier_cloud를 살려서 사용 가능
        cloud_for_clustering = outlier_cloud

        # --------------------
        # 4. HDBSCAN 클러스터링
        # --------------------
        # open3d 포인트클라우드를 numpy array로 변환
        points_np = np.asarray(cloud_for_clustering.points)

        # 만약 포인트가 극단적으로 적으면, 안전하게 예외 처리
        if len(points_np) < 5:
            rospy.logwarn("Not enough points for clustering. Skipping HDBSCAN.")
            return cloud_for_clustering

        # HDBSCAN 파라미터 설정 (예시)
        # - min_cluster_size: 최소 클러스터 크기
        # - metric='euclidean'
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean')
        labels = clusterer.fit_predict(points_np)

        # 클러스터 라벨 확인
        unique_labels = set(labels)
        rospy.loginfo("HDBSCAN clustering finished. Found {} clusters (excluding noise)".format(
            len(unique_labels) - (1 if -1 in unique_labels else 0)
        ))

        # --------------------
        # 5. 객체 BoundingBox 생성
        # --------------------
        # 라벨별로 AxisAlignedBoundingBox를 구해서 예시로 로그로 출력
        # (주로 시각화를 위해서는 MarkerArray 등을 퍼블리시하거나 open3d 시각화 사용)
        for lbl in unique_labels:
            if lbl == -1:
                # -1은 노이즈 처리된 점들
                continue

            cluster_indices = np.where(labels == lbl)[0]
            cluster_points = points_np[cluster_indices]

            # 해당 클러스터 포인트만으로 Open3D PointCloud 생성
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)

            # AxisAlignedBoundingBox 계산
            bbox = cluster_cloud.get_axis_aligned_bounding_box()
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            rospy.loginfo("Cluster {} -> #Points: {}, BBox min:{} max:{}".format(
                lbl, len(cluster_points), min_bound, max_bound))

        # 여기서는 최종적으로 클러스터링 대상이 되었던 점들(outlier_cloud)만 리턴
        # 바닥면(inlier_cloud)을 살리려면 둘을 합쳐서 반환할 수도 있음
        # ex) merged_cloud = inlier_cloud + outlier_cloud
        return cloud_for_clustering

    def convert_ros_to_open3d(self, ros_cloud):
        """
        ROS PointCloud2 메시지를 Open3D 포인트 클라우드로 변환
        """
        points = []
        for p in pc2.read_points(ros_cloud, skip_nans=True):
            points.append([p[0], p[1], p[2]])  # 필요시 intensity, rgb 등을 확장 가능

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(np.array(points))
        return cloud_o3d

    def convert_open3d_to_ros(self, open3d_cloud, header):
        """
        Open3D 포인트 클라우드를 ROS PointCloud2 메시지로 변환
        """
        points = np.asarray(open3d_cloud.points, dtype=np.float32)

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
        ros_cloud.data = points.tobytes()
        return ros_cloud

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        processor.spin()
    except rospy.ROSInternalException:
        pass
