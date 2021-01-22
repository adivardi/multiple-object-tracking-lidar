// #include "kf_tracker/CKalmanFilter.h"
// #include "kf_tracker/featureDetection.h"
#include "opencv2/video/tracking.hpp"
#include "pcl_ros/point_cloud.h"
#include <algorithm>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <string.h>

#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <limits>
#include <utility>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// ------- Adi ------
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/transforms.h>
#include <tf/transform_listener.h>

#include "kf_tracker/KalmanFilterTrack.h"
#include "kf_tracker/TracksManager.h"

using namespace std;
using namespace cv;

ros::Publisher objID_pub;

// KF init
int stateDim = 4; // [x,y,v_x,v_y]//,w,h]
int measDim = 2; // [z_x,z_y,z_w,z_h]
int ctrlDim = 0;
TracksManager tracks_manager;

ros::Publisher pub_cluster0;
ros::Publisher pub_cluster1;
ros::Publisher pub_cluster2;
ros::Publisher pub_cluster3;
ros::Publisher pub_cluster4;
ros::Publisher pub_cluster5;

ros::Publisher markerPub;
ros::Publisher cc_pos;

// std::vector<geometry_msgs::Point> prevClusterCenters;

cv::Mat state(stateDim, 1, CV_32F);
cv::Mat_<float> measurement(2, 1);

std::vector<int> objID; // Output of the data association using KF
                        // measurement.setTo(Scalar(0));

bool firstFrame = true;

// calculate euclidean distance of two points
double
euclidean_distance(geometry_msgs::Point& p1, geometry_msgs::Point& p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}
/*
//Count unique object IDs. just to make sure same ID has not been assigned to two KF_Trackers.
int countIDs(vector<int> v)
{
    transform(v.begin(), v.end(), v.begin(), abs); // O(n) where n = distance(v.end(), v.begin())
    sort(v.begin(), v.end()); // Average case O(n log n), worst case O(n^2) (usually implemented as quicksort.
    // To guarantee worst case O(n log n) replace with make_heap, then sort_heap.

    // Unique will take a sorted range, and move things around to get duplicated
    // items to the back and returns an iterator to the end of the unique section of the range
    auto unique_end = unique(v.begin(), v.end()); // Again n comparisons
    return distance(unique_end, v.begin()); // Constant time for random access iterators (like vector's)
}
*/

/*

objID: vector containing the IDs of the clusters that should be associated with each KF_Tracker
objID[0] corresponds to KFT0, objID[1] corresponds to KFT1 etc.
*/

std::pair<int, int>
findIndexOfMin(std::vector<std::vector<float>> distMat)
{
  // cout << "findIndexOfMin cALLED\n";
  std::pair<int, int> minIndex;
  float minEl = std::numeric_limits<float>::max();
  // cout << "minEl=" << minEl << "\n";
  for (int i = 0; i < distMat.size(); i++)
    for (int j = 0; j < distMat.at(0).size(); j++)
    {
      if (distMat[i][j] < minEl)
      {
        minEl = distMat[i][j];
        minIndex = std::make_pair(i, j);
      }
    }
  // cout << "minIndex=" << minIndex.first << "," << minIndex.second << "\n";
  return minIndex;
}

void
KFT(const std_msgs::Float32MultiArray ccs)
{
  // First predict, to update the internal statePre variable
  std::vector<cv::Mat> predictions = tracks_manager.predict();

  // transform to geo_msgs  // TODO needed?
  std::vector<geometry_msgs::Point> KFpredictions;
  int i = 0;
  for (auto it = predictions.begin(); it != predictions.end(); it++)
  {
    geometry_msgs::Point pt;
    pt.x = (*it).at<float>(0);
    pt.y = (*it).at<float>(1);
    pt.z = (*it).at<float>(2);

    KFpredictions.push_back(pt);
  }

  // cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
  // cout<<"Prediction 1 ="<<prediction.at<float>(0)<<","<<prediction.at<float>(1)<<"\n";

  // Get measurements
  // Extract the position of the clusters from the multiArray. To check if the data
  // coming in, check the .z (every third) coordinate and that will be 0.0
  std::vector<geometry_msgs::Point> clusterCenters; // clusterCenters

  i = 0;
  for (std::vector<float>::const_iterator it = ccs.data.begin(); it != ccs.data.end(); it += 3)
  {
    geometry_msgs::Point pt;
    pt.x = *it;
    pt.y = *(it + 1);
    pt.z = *(it + 2);

    clusterCenters.push_back(pt);
  }

  // Find the cluster that is more probable to be belonging to a given KF.
  objID.clear(); // Clear the objID vector
  objID.resize(6); // Allocate default elements so that [i] doesnt segfault. Should be done better
  // Copy clusterCentres for modifying it and preventing multiple assignments of the same ID
  std::vector<geometry_msgs::Point> copyOfClusterCenters(clusterCenters);
  std::vector<std::vector<float>> distMat;

  for (int filterN = 0; filterN < 6; filterN++)
  {
    std::vector<float> distVec;
    for (int n = 0; n < 6; n++)
    {
      distVec.push_back(euclidean_distance(KFpredictions[filterN], copyOfClusterCenters[n]));
    }

    distMat.push_back(distVec);
    /*// Based on distVec instead of distMat (global min). Has problems with the person's leg going out of scope
       int ID=std::distance(distVec.begin(),min_element(distVec.begin(),distVec.end()));
       //cout<<"finterlN="<<filterN<<"   minID="<<ID
       objID.push_back(ID);
      // Prevent assignment of the same object ID to multiple clusters
       copyOfClusterCenters[ID].x=100000;// A large value so that this center is not assigned to another cluster
       copyOfClusterCenters[ID].y=10000;
       copyOfClusterCenters[ID].z=10000;
      */
    // cout << "filterN=" << filterN << "\n";
  }

  // cout << "distMat.size()" << distMat.size() << "\n";
  // cout << "distMat[0].size()" << distMat.at(0).size() << "\n";
  // DEBUG: print the distMat
  // for (const auto& row : distMat)
  // {
  //   for (const auto& s : row)
  //     std::cout << s << ' ';
  //   std::cout << std::endl;
  // }

  for (int clusterCount = 0; clusterCount < 6; clusterCount++)
  {
    // 1. Find min(distMax)==> (i,j);
    std::pair<int, int> minIndex(findIndexOfMin(distMat));
    // cout << "Received minIndex=" << minIndex.first << "," << minIndex.second << "\n";
    // 2. objID[i]=clusterCenters[j]; counter++
    objID[minIndex.first] = minIndex.second;

    // 3. distMat[i,:]=10000; distMat[:,j]=10000
    distMat[minIndex.first] = std::vector<float>(6, 10000.0); // Set the row to a high number.
    for (int row = 0; row < distMat.size(); row++) // set the column to a high number
    {
      distMat[row][minIndex.second] = 10000.0;
    }
    // 4. if(counter<6) got to 1.
    // cout << "clusterCount=" << clusterCount << "\n";
  }

  // cout<<"Got object IDs"<<"\n";
  // countIDs(objID);// for verif/corner cases

  // display objIDs
  /* DEBUG
    cout<<"objID= ";
    for(auto it=objID.begin();it!=objID.end();it++)
        cout<<*it<<" ,";
    cout<<"\n";
    */

  visualization_msgs::MarkerArray clusterMarkers;

  for (int i = 0; i < 6; i++)
  {
    visualization_msgs::Marker m;

    m.id = i;
    m.type = visualization_msgs::Marker::CUBE;
    m.header.frame_id = "/map";
    m.scale.x = 0.3;
    m.scale.y = 0.3;
    m.scale.z = 0.3;
    m.action = visualization_msgs::Marker::ADD;
    m.color.a = 1.0;
    m.color.r = i % 2 ? 1 : 0;
    m.color.g = i % 3 ? 1 : 0;
    m.color.b = i % 4 ? 1 : 0;

    // geometry_msgs::Point clusterC(clusterCenters.at(objID[i]));
    geometry_msgs::Point clusterC(KFpredictions[i]);
    m.pose.position.x = clusterC.x;
    m.pose.position.y = clusterC.y;
    m.pose.position.z = clusterC.z;

    clusterMarkers.markers.push_back(m);
  }

  // prevClusterCenters = clusterCenters;

  markerPub.publish(clusterMarkers);

  std_msgs::Int32MultiArray obj_id;
  for (auto it = objID.begin(); it != objID.end(); it++)
    obj_id.data.push_back(*it);
  // Publish the object IDs
  objID_pub.publish(obj_id);
  // convert clusterCenters from geometry_msgs::Point to floats
  std::vector<std::vector<float>> cc;
  for (int i = 0; i < 6; i++)
  {
    vector<float> pt;
    pt.push_back(clusterCenters[objID[i]].x);
    pt.push_back(clusterCenters[objID[i]].y);
    pt.push_back(clusterCenters[objID[i]].z);

    cc.push_back(pt);
  }
  // cout<<"cc[5][0]="<<cc[5].at(0)<<"cc[5][1]="<<cc[5].at(1)<<"cc[5][2]="<<cc[5].at(2)<<"\n";
  float meas0[2] = {cc[0].at(0), cc[0].at(1)};
  float meas1[2] = {cc[1].at(0), cc[1].at(1)};
  float meas2[2] = {cc[2].at(0), cc[2].at(1)};
  float meas3[2] = {cc[3].at(0), cc[3].at(1)};
  float meas4[2] = {cc[4].at(0), cc[4].at(1)};
  float meas5[2] = {cc[5].at(0), cc[5].at(1)};

  // The update phase
  cv::Mat meas0Mat = cv::Mat(2, 1, CV_32F, meas0);
  cv::Mat meas1Mat = cv::Mat(2, 1, CV_32F, meas1);
  cv::Mat meas2Mat = cv::Mat(2, 1, CV_32F, meas2);
  cv::Mat meas3Mat = cv::Mat(2, 1, CV_32F, meas3);
  cv::Mat meas4Mat = cv::Mat(2, 1, CV_32F, meas4);
  cv::Mat meas5Mat = cv::Mat(2, 1, CV_32F, meas5);

  // cout<<"meas0Mat"<<meas0Mat<<"\n";
  if (!(meas0Mat.at<float>(0, 0) == 0.0f || meas0Mat.at<float>(1, 0) == 0.0f))
    Mat estimated0 = KF0.correct(meas0Mat);
  if (!(meas1[0] == 0.0f || meas1[1] == 0.0f))
    Mat estimated1 = KF1.correct(meas1Mat);
  if (!(meas2[0] == 0.0f || meas2[1] == 0.0f))
    Mat estimated2 = KF2.correct(meas2Mat);
  if (!(meas3[0] == 0.0f || meas3[1] == 0.0f))
    Mat estimated3 = KF3.correct(meas3Mat);
  if (!(meas4[0] == 0.0f || meas4[1] == 0.0f))
    Mat estimated4 = KF4.correct(meas4Mat);
  if (!(meas5[0] == 0.0f || meas5[1] == 0.0f))
    Mat estimated5 = KF5.correct(meas5Mat);

  Mat statePost = KF0.statePost;
  cout << "statePost 0: " << endl << " " << statePost << endl << endl;

  // Publish the point clouds belonging to each clusters

  // cout<<"estimate="<<estimated.at<float>(0)<<","<<estimated.at<float>(1)<<"\n";
  // Point statePt(estimated.at<float>(0),estimated.at<float>(1));
  // cout<<"DONE KF_TRACKER\n";
}
void
publish_cloud(ros::Publisher& pub, pcl::PointCloud<pcl::PointXYZI>::Ptr cluster)
{
  sensor_msgs::PointCloud2::Ptr clustermsg(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*cluster, *clustermsg);
  clustermsg->header.frame_id = "/base_link";
  clustermsg->header.stamp = ros::Time::now();
  pub.publish(*clustermsg);
}

// ------------------------- Adi ----------------------------
typedef pcl::PointXYZI PointXYZI;
typedef pcl::PointCloud<PointXYZI> PointCloud;
std::string map_frame = "base_link";
std::string base_frame = "base_link";
float voxel_size = 0.05;
float z_min = 0.4;
float z_max = 2.5;
float clustering_tolerance = 0.15;
float min_pts_in_cluster = 50;
ros::Publisher proccessed_pub_;

void
filterGround(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  // TODO can be improved using normals and height
  // currently just cut between z_min and z_max

  pcl::PassThrough<PointXYZI> pass_filter;
  pass_filter.setInputCloud(input);
  pass_filter.setFilterFieldName("z");
  pass_filter.setFilterLimits(z_min, z_max);
  // pass_filter.setFilterLimitsNegative(true);
  pass_filter.filter(*processed);
}

void
processPointCloud(const PointCloud::ConstPtr& input, PointCloud::Ptr& processed)
{
  // remove invalid pts (NaN, Inf)
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*input, *processed, indices);

  // transform_pointcloud to map frame
  tf::TransformListener tf_listener_;
  tf_listener_.waitForTransform(
      processed->header.frame_id, map_frame, fromPCL(processed->header).stamp, ros::Duration(5.0));
  pcl_ros::transformPointCloud<PointXYZI>(map_frame, *processed, *processed, tf_listener_);

  // voxel filter
  pcl::VoxelGrid<PointXYZI> vox_filter;
  vox_filter.setInputCloud(processed);
  vox_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  vox_filter.filter(*processed);

  // remove ground
  filterGround(processed, processed);
}

void
// cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
cloud_cb(const PointCloud::ConstPtr& input_cloud)
{
  PointCloud::Ptr processed_cloud(new PointCloud);
  processPointCloud(input_cloud, processed_cloud);

  // publish processed pointcloud
  proccessed_pub_.publish(processed_cloud->makeShared());

  // Creating the KdTree from input point cloud
  pcl::search::KdTree<PointXYZI>::Ptr tree(new pcl::search::KdTree<PointXYZI>);
  tree->setInputCloud(processed_cloud);

  // clustering
  /* vector of PointIndices, which contains the actual index information in a vector<int>.
  * The indices of each detected cluster are saved here.
  * Cluster_indices is a vector containing one instance of PointIndices for each detected
  * cluster.
  */
  std::vector<pcl::PointIndices> clusters_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> eucl_clustering;
  eucl_clustering.setClusterTolerance(clustering_tolerance);
  eucl_clustering.setMinClusterSize(min_pts_in_cluster);
  // eucl_clustering.setMaxClusterSize(600);
  eucl_clustering.setSearchMethod(tree);
  eucl_clustering.setInputCloud(processed_cloud);
  /* Extract the clusters out of pc and save indices in clusters_indices.*/
  eucl_clustering.extract(clusters_indices);

  std::cout << "cluster nu: " << clusters_indices.size() << std::endl;

  // get cluster centroid
  // std::vector<int>::const_iterator pit;
  // Vector of cluster pointclouds
  std::vector<PointCloud::Ptr> clusters;
  std::vector<Eigen::Vector4f> clusters_centroids;

  for (const pcl::PointIndices& cluster_ids : clusters_indices)
  // for (auto cluster_it = clusters_indices.begin(); cluster_it != clusters_indices.end(); ++cluster_it)
  {
    // Eigen::Matrix4 cluster_centroid;
    Eigen::Vector4f cluster_centroid;
    pcl::compute3DCentroid(*processed_cloud, cluster_ids, cluster_centroid); // in homogenous coords
    cout << "centroid: " << cluster_centroid << endl;

    clusters_centroids.push_back(cluster_centroid);

    // Extract the cluster pointcloud
    PointCloud::Ptr cluster_pointcloud(new PointCloud);
    pcl::copyPointCloud(*processed_cloud, clusters_indices, *cluster_pointcloud);

    clusters.push_back(cluster_pointcloud);
  }

  // TODO get rid of this crap
  // Ensure at least 6 clusters exist to publish (later clusters may be empty)
  while (clusters.size() < 6)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr empty_cluster(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointXYZI pt;
    empty_cluster->points.push_back(pt);
    clusters.push_back(empty_cluster);
  }

  while (clusters_centroids.size() < 6)
  {
    Eigen::Vector4f centroid;
    centroid[0] = 0.0;
    centroid[1] = 0.0;
    centroid[2] = 0.0;

    clusters_centroids.push_back(centroid);
  }

  // Publish cluster mid-points.
  std_msgs::Float32MultiArray cc;
  for (int i = 0; i < 6; i++)
  {
    cc.data.push_back(clusters_centroids.at(i)[0]);
    cc.data.push_back(clusters_centroids.at(i)[1]);
    cc.data.push_back(clusters_centroids.at(i)[2]);
  }

  cc_pos.publish(cc);

  // cout<<"IF firstFrame="<<firstFrame<<"\n";
  // If this is the first frame, initialize kalman filters for the clustered objects
  if (firstFrame)
  {
    // Initialize 6 Kalman Filters; Assuming 6 max objects in the dataset.
    // Could be made generic by creating a Kalman Filter only when a new object is detected

    // float dvx = 0.01f; // 1.0
    // float dvy = 0.01f; // 1.0
    // float dx = 1.0f;
    // float dy = 1.0f;
    // KF0.transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);
    // KF1.transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);
    // KF2.transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);
    // KF3.transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);
    // KF4.transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);
    // KF5.transitionMatrix = (Mat_<float>(4, 4) << dx, 0, 1, 0, 0, dy, 0, 1, 0, 0, dvx, 0, 0, 0, 0, dvy);

    // cv::setIdentity(KF0.measurementMatrix);
    // cv::setIdentity(KF1.measurementMatrix);
    // cv::setIdentity(KF2.measurementMatrix);
    // cv::setIdentity(KF3.measurementMatrix);
    // cv::setIdentity(KF4.measurementMatrix);
    // cv::setIdentity(KF5.measurementMatrix);

    // // Process Noise Covariance Matrix Q
    // // [ Ex 0  0    0 0    0 ]
    // // [ 0  Ey 0    0 0    0 ]
    // // [ 0  0  Ev_x 0 0    0 ]
    // // [ 0  0  0    1 Ev_y 0 ]
    // //// [ 0  0  0    0 1    Ew ]
    // //// [ 0  0  0    0 0    Eh ]
    // float sigmaP = 0.01;
    // float sigmaQ = 0.1;
    // setIdentity(KF0.processNoiseCov, Scalar::all(sigmaP));
    // setIdentity(KF1.processNoiseCov, Scalar::all(sigmaP));
    // setIdentity(KF2.processNoiseCov, Scalar::all(sigmaP));
    // setIdentity(KF3.processNoiseCov, Scalar::all(sigmaP));
    // setIdentity(KF4.processNoiseCov, Scalar::all(sigmaP));
    // setIdentity(KF5.processNoiseCov, Scalar::all(sigmaP));
    // // Meas noise cov matrix R
    // cv::setIdentity(KF0.measurementNoiseCov, cv::Scalar(sigmaQ)); // 1e-1
    // cv::setIdentity(KF1.measurementNoiseCov, cv::Scalar(sigmaQ));
    // cv::setIdentity(KF2.measurementNoiseCov, cv::Scalar(sigmaQ));
    // cv::setIdentity(KF3.measurementNoiseCov, cv::Scalar(sigmaQ));
    // cv::setIdentity(KF4.measurementNoiseCov, cv::Scalar(sigmaQ));
    // cv::setIdentity(KF5.measurementNoiseCov, cv::Scalar(sigmaQ));

    // // Set initial state
    // KF0.statePre.at<float>(0) = clusters_centroids.at(0)[0];
    // KF0.statePre.at<float>(1) = clusters_centroids.at(0)[1];
    // KF0.statePre.at<float>(2) = 0; // initial v_x
    // KF0.statePre.at<float>(3) = 0; // initial v_y

    // // Set initial state
    // KF1.statePre.at<float>(0) = clusters_centroids.at(1)[0];
    // KF1.statePre.at<float>(1) = clusters_centroids.at(1)[1];
    // KF1.statePre.at<float>(2) = 0; // initial v_x
    // KF1.statePre.at<float>(3) = 0; // initial v_y

    // // Set initial state
    // KF2.statePre.at<float>(0) = clusters_centroids.at(2)[0];
    // KF2.statePre.at<float>(1) = clusters_centroids.at(2)[1];
    // KF2.statePre.at<float>(2) = 0; // initial v_x
    // KF2.statePre.at<float>(3) = 0; // initial v_y

    // // Set initial state
    // KF3.statePre.at<float>(0) = clusters_centroids.at(3)[0];
    // KF3.statePre.at<float>(1) = clusters_centroids.at(3)[1];
    // KF3.statePre.at<float>(2) = 0; // initial v_x
    // KF3.statePre.at<float>(3) = 0; // initial v_y

    // // Set initial state
    // KF4.statePre.at<float>(0) = clusters_centroids.at(4)[0];
    // KF4.statePre.at<float>(1) = clusters_centroids.at(4)[1];
    // KF4.statePre.at<float>(2) = 0; // initial v_x
    // KF4.statePre.at<float>(3) = 0; // initial v_y

    // // Set initial state
    // KF5.statePre.at<float>(0) = clusters_centroids.at(5)[0];
    // KF5.statePre.at<float>(1) = clusters_centroids.at(5)[1];
    // KF5.statePre.at<float>(2) = 0; // initial v_x
    // KF5.statePre.at<float>(3) = 0; // initial v_y

    for (const auto& centroid : clusters_centroids)
    {
      tracks_manager.add_track(centroid);
    }
    firstFrame = false;

    // for (int i = 0; i < 6; i++)
    // {
    //   geometry_msgs::Point pt;
    //   pt.x = clusters_centroids.at(i)[0];
    //   pt.y = clusters_centroids.at(i)[1];
    //   prevClusterCenters.push_back(pt);
    // }
    /*  // Print the initial state of the kalman filter for debugging
     cout<<"KF0.satePre="<<KF0.statePre.at<float>(0)<<","<<KF0.statePre.at<float>(1)<<"\n";
     cout<<"KF1.satePre="<<KF1.statePre.at<float>(0)<<","<<KF1.statePre.at<float>(1)<<"\n";
     cout<<"KF2.satePre="<<KF2.statePre.at<float>(0)<<","<<KF2.statePre.at<float>(1)<<"\n";
     cout<<"KF3.satePre="<<KF3.statePre.at<float>(0)<<","<<KF3.statePre.at<float>(1)<<"\n";
     cout<<"KF4.satePre="<<KF4.statePre.at<float>(0)<<","<<KF4.statePre.at<float>(1)<<"\n";
     cout<<"KF5.satePre="<<KF5.statePre.at<float>(0)<<","<<KF5.statePre.at<float>(1)<<"\n";

     //cin.ignore();// To be able to see the printed initial state of the KalmanFilter
     */
  }

  else
  {
    // cout<<"ELSE firstFrame="<<firstFrame<<"\n";
    KFT(cc);

    int i = 0;
    bool publishedCluster[6];
    for (auto it = objID.begin(); it != objID.end(); it++)
    {
      switch (i)
      {
        case 0:
        {
          publish_cloud(pub_cluster0, clusters[*it]);
          publishedCluster[i] = true; // Use this flag to publish only once for a given obj ID
          i++;
          break;
        }
        case 1:
        {
          publish_cloud(pub_cluster1, clusters[*it]);
          publishedCluster[i] = true; // Use this flag to publish only once for a given obj ID
          i++;
          break;
        }
        case 2:
        {
          publish_cloud(pub_cluster2, clusters[*it]);
          publishedCluster[i] = true; // Use this flag to publish only once for a given obj ID
          i++;
          break;
        }
        case 3:
        {
          publish_cloud(pub_cluster3, clusters[*it]);
          publishedCluster[i] = true; // Use this flag to publish only once for a given obj ID
          i++;
          break;
        }
        case 4:
        {
          publish_cloud(pub_cluster4, clusters[*it]);
          publishedCluster[i] = true; // Use this flag to publish only once for a given obj ID
          i++;
          break;
        }

        case 5:
        {
          publish_cloud(pub_cluster5, clusters[*it]);
          publishedCluster[i] = true; // Use this flag to publish only once for a given obj ID
          i++;
          break;
        }
        default:
          break;
      }
    }
  }
}

int
main(int argc, char** argv)
{
  // ROS init
  ros::init(argc, argv, "kf_tracker");
  ros::NodeHandle nh;

  // Publishers to publish the state of the objects (pos and vel)
  // objState1=nh.advertise<geometry_msgs::Twist> ("obj_1",1);

  cout << "About to setup callback\n";

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("filtered_cloud", 1, cloud_cb);
  // Create a ROS publisher for the output point cloud
  pub_cluster0 = nh.advertise<sensor_msgs::PointCloud2>("cluster_0", 1);
  pub_cluster1 = nh.advertise<sensor_msgs::PointCloud2>("cluster_1", 1);
  pub_cluster2 = nh.advertise<sensor_msgs::PointCloud2>("cluster_2", 1);
  pub_cluster3 = nh.advertise<sensor_msgs::PointCloud2>("cluster_3", 1);
  pub_cluster4 = nh.advertise<sensor_msgs::PointCloud2>("cluster_4", 1);
  pub_cluster5 = nh.advertise<sensor_msgs::PointCloud2>("cluster_5", 1);
  // Subscribe to the clustered pointclouds
  // ros::Subscriber c1=nh.subscribe("ccs",100,KFT);
  objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id", 1);
  /* Point cloud clustering
   */

  cc_pos = nh.advertise<std_msgs::Float32MultiArray>("ccs", 100); // clusterCenter1
  markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz", 1);

  proccessed_pub_ = nh.advertise<PointCloud>("processed", 1);
  /* Point cloud clustering
   */

  ros::spin();
}
