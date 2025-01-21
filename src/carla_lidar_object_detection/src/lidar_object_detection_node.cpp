#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class LidarObjectDetector
{
public:
    LidarObjectDetector(ros::NodeHandle& nh)
      : nh_(nh),
        tf_listener_(tf_buffer_)
    {
        // LiDAR 포인트 클라우드 구독
        pointcloud_sub_ = nh_.subscribe("/carla/ego_vehicle/lidar", 1, &LidarObjectDetector::pointCloudCallback, this);

        // 객체 검출 결과 마커 퍼블리셔
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/detected_objects", 1);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher marker_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        // PCL 변환
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // Voxel Grid Downsampling
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.2f, 0.2f, 0.2f);
        vg.filter(*cloud_filtered);

        if (cloud_filtered->empty()) {
            ROS_WARN("Empty filtered cloud");
            return;
        }

        // 클러스터링
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(1.0);
        ec.setMinClusterSize(30);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        visualization_msgs::MarkerArray marker_array;
        int id = 0;

        for (auto &indices : cluster_indices) {
            // 각 클러스터의 min/max point 계산
            float min_x = std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float min_z = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();

            for (auto &idx : indices.indices) {
                pcl::PointXYZ pt = cloud_filtered->points[idx];
                if (pt.x < min_x) min_x = pt.x;
                if (pt.y < min_y) min_y = pt.y;
                if (pt.z < min_z) min_z = pt.z;
                if (pt.x > max_x) max_x = pt.x;
                if (pt.y > max_y) max_y = pt.y;
                if (pt.z > max_z) max_z = pt.z;
            }

            float box_length = max_x - min_x;
            float box_width = max_y - min_y;
            float box_height = max_z - min_z;

            // 박스 중심
            float cx = (max_x + min_x) / 2.0;
            float cy = (max_y + min_y) / 2.0;
            float cz = (max_z + min_z) / 2.0;

            // Marker 생성
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map"; // 여기서는 base_link 기준, 필요시 tf 변환 가능
            marker.header.stamp = ros::Time::now();
            marker.ns = "detected_objects";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = cx;
            marker.pose.position.y = cy;
            marker.pose.position.z = cz;

            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = box_length;
            marker.scale.y = box_width;
            marker.scale.z = box_height;

            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.5;

            marker.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(marker);
        }

        marker_pub_.publish(marker_array);
        ROS_INFO("Detected %lu clusters and published bounding boxes.", cluster_indices.size());
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_object_detection_node");
    ros::NodeHandle nh;

    LidarObjectDetector detector(nh);

    ros::spin();
    return 0;
}
