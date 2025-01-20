#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

class PointCloudProcessor {
public:
    PointCloudProcessor(ros::NodeHandle& nh) {
        // ROS 토픽 구독 및 발행 설정
        sub_ = nh.subscribe("/velodyne_points", 10, &PointCloudProcessor::callback, this);
        cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/processed_points", 10);
        marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/processed_bboxes", 10);
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        // PointCloud2 → PCL 포인트 클라우드 변환
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // 다운샘플링
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.0001, 0.0001, 0.0001);
        vg.filter(*cloud_downsampled);

        // 평면 분할 (RANSAC)
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.3);
        seg.setInputCloud(cloud_downsampled);
        seg.segment(*inliers, *coefficients);

        // RANSAC 결과로 평면 제거
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(cloud_downsampled);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_filtered);

        // 클러스터링 (Euclidean Cluster Extraction)
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.2);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(300);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        // 바운딩 박스 생성 및 발행
        visualization_msgs::MarkerArray marker_array;
        int id = 0;
        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (int index : indices.indices) {
                cluster->points.push_back(cloud_filtered->points[index]);
            }

            // getMinMax3D 호출 수정
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            // 바운딩 박스 중심 및 크기 계산
            float center_x = (min_pt[0] + max_pt[0]) / 2.0f;
            float center_y = (min_pt[1] + max_pt[1]) / 2.0f;
            float center_z = (min_pt[2] + max_pt[2]) / 2.0f;
            float size_x = max_pt[0] - min_pt[0];
            float size_y = max_pt[1] - min_pt[1];
            float size_z = max_pt[2] - min_pt[2];

            // Marker 생성
            visualization_msgs::Marker marker;
            marker.header.frame_id = "velodyne";
            marker.header.stamp = ros::Time::now();
            marker.ns = "bboxes";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = center_x;
            marker.pose.position.y = center_y;
            marker.pose.position.z = center_z;
            marker.scale.x = size_x;
            marker.scale.y = size_y;
            marker.scale.z = size_z;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.2;
            marker_array.markers.push_back(marker);
        }
        marker_pub_.publish(marker_array);

        // PCL → PointCloud2 변환 및 발행
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_filtered, output);
        output.header = msg->header;
        cloud_pub_.publish(output);
    }

private:
    ros::Subscriber sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher marker_pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_processor");
    ros::NodeHandle nh;
    PointCloudProcessor processor(nh);
    ros::spin();
    return 0;
}
