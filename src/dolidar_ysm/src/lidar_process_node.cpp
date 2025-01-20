/*
 * Example: ROS + Open3D(C++ API) + DBSCAN (대안) for clustering
 * 
 * 빌드 시:
 *  - CMakeLists.txt에서 Open3D, ROS 관련 설정 필요
 *  - (HDBSCAN C++ 라이브러리 사용 시) 별도의 라이브러리 링크 필요
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Header.h>

#include <open3d/Open3D.h>
#include <Eigen/Dense>

// (참고) C++ HDBSCAN 라이브러리를 사용하려면 별도의 외부 라이브러리를 찾아서 include해야 함.
// #include <hdbscan-cpp/some_header.h> // 가상의 예시

class PointCloudProcessor {
public:
    PointCloudProcessor() {
        ros::NodeHandle nh("~");

        // Subscriber & Publisher
        sub_ = nh.subscribe("/velodyne_points", 1,
                            &PointCloudProcessor::pointcloudCallback, this);
        pub_ = nh.advertise<sensor_msgs::PointCloud2>("/processed_points", 1);

        ROS_INFO("PointCloudProcessor node initialized.");
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        ROS_INFO("Received PointCloud2 data.");

        // 1. Convert ROS -> Open3D
        auto cloud_o3d = rosToOpen3D(*msg);

        // 2. Processing pipeline
        auto processed_o3d = processPointCloud(cloud_o3d);

        // 3. Convert back to ROS
        sensor_msgs::PointCloud2 output_msg;
        open3DToRos(processed_o3d, msg->header, output_msg);

        // 4. Publish
        pub_.publish(output_msg);
        ROS_INFO("Published processed PointCloud2 data.");
    }

    open3d::geometry::PointCloud processPointCloud(
            const open3d::geometry::PointCloud &input_cloud) {

        // Clone input
        open3d::geometry::PointCloud cloud_o3d = input_cloud;

        // 1. Voxel Downsampling
        double voxel_size = 0.1;
        cloud_o3d = *cloud_o3d.VoxelDownSample(voxel_size);
        ROS_INFO("Downsampled: %lu points -> %lu points",
                 input_cloud.points_.size(), cloud_o3d.points_.size());

        // 2. Outlier Removal (StatisticalOutlierRemoval)
        //    Open3D C++에는 CreateFrom... 형태의 정적 메서드가 있거나,
        //    직접 outlier 인덱스를 얻어서 제거할 수 있습니다.
        {
            size_t nb_neighbors = 20;
            double std_ratio = 2.0;
            std::vector<size_t> inliers;
            std::shared_ptr<open3d::geometry::PointCloud> cl;
            std::tie(cl, inliers) = cloud_o3d.RemoveStatisticalOutlier(
                    nb_neighbors, std_ratio);
            ROS_INFO("Outlier removal: %lu -> %lu points",
                     cloud_o3d.points_.size(), cl->points_.size());
            cloud_o3d = *cl; // update
        }

        // 3. RANSAC Plane Segmentation
        {
            double distance_threshold = 0.3;
            int ransac_n = 3;
            int num_iterations = 1000;

            // plane_model: (a, b, c, d)
            // inliers: 평면에 속한 포인트 인덱스
            Eigen::Vector4d plane_model;
            std::vector<size_t> inliers;
            std::tie(plane_model, inliers) =
                cloud_o3d.SegmentPlane(distance_threshold, ransac_n, num_iterations);

            // 평면(inlier) / 나머지(outlier)로 분리
            auto inlier_cloud = cloud_o3d.SelectByIndex(inliers);
            auto outlier_cloud = cloud_o3d.SelectByIndex(inliers, /*invert=*/true);

            ROS_INFO("Plane segmentation: plane_pts=%lu, others=%lu",
                     inlier_cloud->points_.size(), outlier_cloud->points_.size());

            // 예: 바닥면 제거. outlier_cloud만 후속 처리
            cloud_o3d = *outlier_cloud;
        }

        // 4. HDBSCAN (또는 DBSCAN) 클러스터링
        //    - Open3D는 C++ API에서 DBSCAN을 제공.
        //    - HDBSCAN C++ 구현은 별도 라이브러리 필요.
        {
            // (1) DBSCAN 예시 (Open3D 기본 제공)
            double eps = 0.5;      // 클러스터 거리 임계값
            size_t min_points = 30; // 클러스터 최소 포인트 수
            std::vector<int> labels = cloud_o3d.ClusterDBSCAN(eps, min_points, false);

            // (HDBSCAN을 쓰고 싶다면, 별도 라이브러리를 사용하여 구현해야 합니다.)
            // e.g. pseudo-code for HDBSCAN:
            // hdbscan::HDBSCAN clusterer(...);
            // std::vector<int> labels = clusterer.run(points);

            // 클러스터 수 파악
            std::set<int> unique_labels(labels.begin(), labels.end());
            size_t noise_count = 0;
            for (auto &lbl : unique_labels) {
                if (lbl < 0) noise_count++;
            }
            ROS_INFO("Clustering finished. Found %lu cluster(s) + noise(%lu)",
                     unique_labels.size() - noise_count, noise_count);

            // 5. 클러스터별 BoundingBox (AABB)
            //    label 별로 포인트를 추출 -> AABB 계산
            std::vector<Eigen::Vector3d> points = cloud_o3d.points_;
            for (auto lbl : unique_labels) {
                if (lbl < 0) {
                    // noise
                    continue;
                }
                // lbl에 해당하는 포인트만 모으기
                std::vector<Eigen::Vector3d> cluster_points;
                cluster_points.reserve(points.size());
                for (size_t i = 0; i < labels.size(); ++i) {
                    if (labels[i] == lbl) {
                        cluster_points.push_back(points[i]);
                    }
                }
                // Open3D PointCloud 생성
                open3d::geometry::PointCloud cluster_cloud;
                cluster_cloud.points_ = cluster_points;

                // AxisAlignedBoundingBox 계산
                auto aabb = cluster_cloud.GetAxisAlignedBoundingBox();
                auto min_bound = aabb.min_bound_;
                auto max_bound = aabb.max_bound_;
                ROS_INFO("Cluster %d -> #Points: %lu, BBox min:[%.2f %.2f %.2f], max:[%.2f %.2f %.2f]",
                         lbl, cluster_points.size(),
                         min_bound.x(), min_bound.y(), min_bound.z(),
                         max_bound.x(), max_bound.y(), max_bound.z());
            }
        }

        // 여기서는 최종적으로 클러스터링 대상 포인트(= plane 제거 후 남은 포인트)를 반환
        return cloud_o3d;
    }

    open3d::geometry::PointCloud rosToOpen3D(const sensor_msgs::PointCloud2 &msg) {
        // XYZ만 사용한다고 가정 (fields: x, y, z).
        open3d::geometry::PointCloud cloud_o3d;
        
        // ROS의 PointCloud2 -> raw data 접근
        // 각 point_step = 12 bytes (x, y, z : float32)
        // 단, 실제 메시지 필드 구성이 다를 수도 있으니 주의
        int offset_x = 0;
        int offset_y = 4;
        int offset_z = 8;
        int point_step = msg.point_step;

        size_t num_points = msg.width * msg.height;
        cloud_o3d.points_.reserve(num_points);

        const uint8_t* data_ptr = &msg.data[0];
        for (size_t i = 0; i < num_points; ++i) {
            float px = *reinterpret_cast<const float*>(data_ptr + offset_x);
            float py = *reinterpret_cast<const float*>(data_ptr + offset_y);
            float pz = *reinterpret_cast<const float*>(data_ptr + offset_z);

            // NaN 체크
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
                data_ptr += point_step;
                continue;
            }

            cloud_o3d.points_.push_back(Eigen::Vector3d(px, py, pz));

            data_ptr += point_step;
        }

        return cloud_o3d;
    }

    void open3DToRos(const open3d::geometry::PointCloud &cloud_o3d,
                     const std_msgs::Header &header,
                     sensor_msgs::PointCloud2 &msg_out) {
        // XYZ만 출력
        msg_out.header = header;
        msg_out.height = 1;
        msg_out.width = static_cast<uint32_t>(cloud_o3d.points_.size());

        msg_out.fields.resize(3);
        msg_out.fields[0].name = "x";
        msg_out.fields[0].offset = 0;
        msg_out.fields[0].count = 1;
        msg_out.fields[0].datatype = sensor_msgs::PointField::FLOAT32;

        msg_out.fields[1].name = "y";
        msg_out.fields[1].offset = 4;
        msg_out.fields[1].count = 1;
        msg_out.fields[1].datatype = sensor_msgs::PointField::FLOAT32;

        msg_out.fields[2].name = "z";
        msg_out.fields[2].offset = 8;
        msg_out.fields[2].count = 1;
        msg_out.fields[2].datatype = sensor_msgs::PointField::FLOAT32;

        msg_out.is_bigendian = false;
        msg_out.point_step = 12;  // 3 * 4 bytes
        msg_out.row_step = msg_out.point_step * msg_out.width;
        msg_out.is_dense = true;

        // 데이터 채우기
        msg_out.data.resize(msg_out.row_step);

        // copy points
        uint8_t* data_ptr = &msg_out.data[0];
        for (size_t i = 0; i < cloud_o3d.points_.size(); ++i) {
            float x = static_cast<float>(cloud_o3d.points_[i].x());
            float y = static_cast<float>(cloud_o3d.points_[i].y());
            float z = static_cast<float>(cloud_o3d.points_[i].z());

            memcpy(data_ptr + 0, &x, sizeof(float));
            memcpy(data_ptr + 4, &y, sizeof(float));
            memcpy(data_ptr + 8, &z, sizeof(float));

            data_ptr += msg_out.point_step;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_process_node_cpp");
    PointCloudProcessor processor;
    ros::spin();
    return 0;
}
