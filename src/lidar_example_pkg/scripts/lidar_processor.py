import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import MarkerArray, Marker
import open3d as o3d
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import hdbscan



# ROS callback function
def callback(msg):
    
    # subscribe lidar's binary data
    pcd = binary_to_open3d(msg)
    bbox_objects = process_pointcloud(pcd)

    # publish bounding box
    marker_array = create_marker_array(bbox_objects)
    pub.publish(marker_array)

# binary data from lidar -> open3d
def binary_to_open3d(msg):
    
    # convert binary into numpy
    points = []
    for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2]])
    
    # convert numpy into open3d object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points))
    return cloud


def RANSAC(pcd, distance_threshold, num_iterations):
    degrees = [1, 2] # both 1 and 2
    max_inliers = [] # inliers of the model(plane or surface) whose the number of inliers is bigger than the other
    max_inliers_degree = None # degree of the model (plane or surface) whose the number of inliers is bigger than the other
    max_coefficients = None  # the coeff of the model (plane or surface) whose the number of inliers is bigger than the other

    for degree in degrees:
        pcd_points = np.array(pcd.points) # n*3 array
        X_Y = pcd_points[:, :2] # n*2 array
        Z = pcd_points[:, 2] # n*1 array
        
        # Polynomial features for quadratic model
        poly_features = PolynomialFeatures(degree=degree, include_bias=True) # set degree
        X_poly = poly_features.fit_transform(X_Y) # fitting 
        
        
        ransac = RANSACRegressor(min_samples=3*degree, residual_threshold= distance_threshold, max_trials=num_iterations) # if degree is 1 then min_smaple is 3. if degree is 2 then min_smaple is 6
        ransac.fit(X_poly, Z) # training with RANSAC
        
        
        inlier_mask = ransac.inlier_mask_ # inliners came from training with RANSAC
        inliers = np.nonzero(inlier_mask)[0] # then find the indices of inliers
        
        # 1degree vs 2degree
        if len(max_inliers) < len(inliers): 
            max_inliers = inliers
            max_inliers_degree = degree
            max_coefficients = ransac.estimator_.coef_
            print(f"max_inliers_degree: {max_inliers_degree}")

    return max_inliers



def process_pointcloud(pcd):
    
    # voxel grid downsampling
    pcd = pcd.voxel_down_sample(voxel_size=0.5)
    
    
    # RANSAC
    inliers = RANSAC(pcd, distance_threshold = 0.3, num_iterations = 500)
    # road_cloud = pcd.select_by_index(inliers)
    pcd = pcd.select_by_index(inliers, invert=True)
    
    
    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
    clusterer.fit(np.array(pcd.points))
    labels = clusterer.labels_ # 1*n numpy array
    
    # numpy lables into python labels
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

    # 3D bounding box
    MAX_POINTS = 300 # max points for each bb
    MIN_POINTS = 50 # min points for each bb
    bbox_objects = []
    for i in range(0, len(indexes)): #len(indexes) : the number of cluster
        
        nb_points = len(pcd.select_by_index(indexes[i]).points) # find the number of 'specific cluster's points'
        if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
            
            sub_cloud = pcd.select_by_index(indexes[i]) # specific cluster
            
            bbox_object = sub_cloud.get_axis_aligned_bounding_box()
            bbox_objects.append(bbox_object)

    return bbox_objects


# ROS publisher for bounding box visualization
def create_marker_array(bbox_objects):
    marker_array = MarkerArray()
    for i, bbox in enumerate(bbox_objects):
        
        marker = Marker() # message type
        
        marker.header.frame_id = "velodyne" # coordinate name
        
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = bbox.get_center()[0]
        marker.pose.position.y = bbox.get_center()[1]
        marker.pose.position.z = bbox.get_center()[2]
        
        marker.scale.x = bbox.get_extent()[0]
        marker.scale.y = bbox.get_extent()[1]
        marker.scale.z = bbox.get_extent()[2]
        
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1 # alpha
        
        marker.id = i
        marker_array.markers.append(marker)
    return marker_array



# ROS node setting
if __name__ == "__main__":
    
    # node name
    rospy.init_node('lidar_processor', anonymous=True) 
    
    # rospy.Subscriber : subscrbe ROS topic
    # /velodyne_points : lidar's topic name
    # PointCloud2 : data type of lidar's topic
    # callback : function to be called when a new message arrives
    rospy.Subscriber('/velodyne_points', PointCloud2, callback)
    
    # rospy.Publisher : publish ROS topic
    # /processed_bboxes : topic name to publish bounding box data
    # MarkerArray : data type to publish bounding box data
    # queue_size : maximum size of queue. good with small value in real time task.
    pub = rospy.Publisher('/processed_bboxes', MarkerArray, queue_size=10)

    # rospy.spin() : keep the node running until it is explicitly terminated with Ctrl+C
    rospy.spin()
