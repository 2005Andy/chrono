#ifndef CHRONO_CHUTILSMESHTOPOINT_H
#define CHRONO_CHUTILSMESHTOPOINT_H

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/distances.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/console/time.h> // TicToc
#include <pcl/ml/kmeans.h>

namespace chrono{
namespace fsi{
namespace utils{

pcl::PolygonMesh::Ptr loadOBJFileToPolygonMesh(const std::string& filename);

pcl::PolygonMesh::Ptr convertFbxToPclMesh(const std::string& fbxFilePath);

void samplePointCloud(const pcl::PolygonMesh::Ptr& mesh, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int num_samples);
void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, float voxel_leaf_size);
void filterPointCloud2(const pcl::PCLPointCloud2::Ptr& cloud_in, pcl::PCLPointCloud2::Ptr& cloud_filtered, float radius);
void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, int num_samples);
void com_dis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
void savePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& txt_filename);
void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
void generateMultiLayerPointCloudWithKMeansPlusPlus(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
                                                    int num_layers, float scale_factor, int num_clusters);
void MeshToPoint(int method, const std::string input_filename, const std::string output_filename, int layer, float voxel,
                 int clusters=1, float rate=0.99, int first_num_samples = 1000000);

}
}
}

#endif  // CHRONO_CHUTILSMESHTOPOINT_H
