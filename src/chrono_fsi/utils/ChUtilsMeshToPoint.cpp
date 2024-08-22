#include "chrono_fsi/utils/ChUtilsMeshToPoint.h"

namespace chrono {
namespace fsi {
namespace utils {

std::ofstream log_file("log.txt");

void log(const std::string& message) {
    log_file << message << std::endl;
    std::cout << message << std::endl;
}


void MeshToPoint(int method, const std::string input_filename, const std::string output_filename, int layer, float voxel,
                 int clusters, float rate, int first_num_samples){
    auto mesh = loadOBJFileToPolygonMesh(input_filename);

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    samplePointCloud(mesh, cloud, first_num_samples);
    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    filterPointCloud(cloud, cloud_filtered, voxel);
    com_dis(cloud_filtered);
    auto cloud_final = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    generateMultiLayerPointCloudWithKMeansPlusPlus(cloud_filtered, cloud_final, layer, rate, clusters);
    savePointCloud(cloud_final, output_filename);
    log_file.close();
}


// 将非三角形面拆分为三角形面
void triangulateFace(const aiFace& face, std::vector<pcl::Vertices>& polygons) {
    if (face.mNumIndices == 3) {
        pcl::Vertices tri;
        tri.vertices.assign(face.mIndices, face.mIndices + 3);
        polygons.push_back(tri);
    } else {
        for (unsigned int i = 1; i < face.mNumIndices - 1; ++i) {
            pcl::Vertices tri;
            tri.vertices.push_back(face.mIndices[0]);
            tri.vertices.push_back(face.mIndices[i]);
            tri.vertices.push_back(face.mIndices[i + 1]);
            polygons.push_back(tri);
        }
    }
}

// 加载OBJ文件并生成pcl::PolygonMesh::Ptr
pcl::PolygonMesh::Ptr loadOBJFileToPolygonMesh(const std::string& filename) {
    // 使用Assimp加载OBJ文件
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate);
    if (!scene) {
        std::cerr << "Error loading file: " << importer.GetErrorString() << std::endl;
        return nullptr;
    }

    // 创建网格对象
    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh* ai_mesh = scene->mMeshes[i];

        // 顶点信息
        for (unsigned int v = 0; v < ai_mesh->mNumVertices; ++v) {
            pcl::PointXYZ point;
            point.x = ai_mesh->mVertices[v].x;
            point.y = ai_mesh->mVertices[v].y;
            point.z = ai_mesh->mVertices[v].z;
            cloud->points.push_back(point);
        }

        // 面信息，确保所有面都是三角面
        for (unsigned int f = 0; f < ai_mesh->mNumFaces; ++f) {
            const aiFace& face = ai_mesh->mFaces[f];
            triangulateFace(face, mesh->polygons);
        }
    }

    // 将点云数据转换为PCL格式
    pcl::toPCLPointCloud2(*cloud, mesh->cloud);

    return mesh;
}

// max文件到obj
pcl::PolygonMesh::Ptr convertFbxToPclMesh(const std::string& fbxFilePath) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(fbxFilePath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
    if (!scene) {
        std::cerr << "Error importing file: " << importer.GetErrorString() << std::endl;
        return nullptr;
    }

    pcl::PolygonMesh::Ptr pclMesh(new pcl::PolygonMesh);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];

        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            pcl::PointXYZ point;
            point.x = mesh->mVertices[v].x;
            point.y = mesh->mVertices[v].y;
            point.z = mesh->mVertices[v].z;
            cloud->points.push_back(point);
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            pcl::Vertices vertices;
            vertices.vertices.resize(face.mNumIndices);
            for (unsigned int i = 0; i < face.mNumIndices; ++i) {
                vertices.vertices[i] = face.mIndices[i];
            }
            pclMesh->polygons.push_back(vertices);
        }
    }

    // 将点云转换为 pcl::PCLPointCloud2 格式
    pcl::toPCLPointCloud2(*cloud, pclMesh->cloud);

    return pclMesh;
}


// 从三角网格采样生成点云
void samplePointCloud(const pcl::PolygonMesh::Ptr& mesh, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int num_samples) {

    // 从PolygonMesh中获取点云数据
    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
    pcl::fromPCLPointCloud2(mesh->cloud, temp_cloud);

    // 计算每个三角形的面积和累积面积
    std::vector<double> cumulative_areas(mesh->polygons.size(), 0.0);
    double total_area = 0.0;
    for (size_t i = 0; i < mesh->polygons.size(); ++i) {
        const auto& polygon = mesh->polygons[i];
        if (polygon.vertices.size() != 3) {
            continue; // 仅处理三角面
        }

        Eigen::Vector3f v1 = temp_cloud.points[polygon.vertices[0]].getVector3fMap();
        Eigen::Vector3f v2 = temp_cloud.points[polygon.vertices[1]].getVector3fMap();
        Eigen::Vector3f v3 = temp_cloud.points[polygon.vertices[2]].getVector3fMap();

        double area = 0.5 * ((v2 - v1).cross(v3 - v1)).norm();
        total_area += area;
        cumulative_areas[i] = total_area;
    }

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, total_area);

    // 采样
    pcl::PointCloud<pcl::PointXYZ> sampled_cloud;
    for (int i = 0; i < num_samples; ++i) {
        double r = dis(gen);
        auto it = std::lower_bound(cumulative_areas.begin(), cumulative_areas.end(), r);
        size_t index = std::distance(cumulative_areas.begin(), it);

        if (index >= mesh->polygons.size()) {
            continue;
        }

        const auto& polygon = mesh->polygons[index];
        Eigen::Vector3f v1 = temp_cloud.points[polygon.vertices[0]].getVector3fMap();
        Eigen::Vector3f v2 = temp_cloud.points[polygon.vertices[1]].getVector3fMap();
        Eigen::Vector3f v3 = temp_cloud.points[polygon.vertices[2]].getVector3fMap();

        float r1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float r2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (r1 + r2 > 1.0f) {
            r1 = 1.0f - r1;
            r2 = 1.0f - r2;
        }

        Eigen::Vector3f sampled_point = v1 + r1 * (v2 - v1) + r2 * (v3 - v1);
        sampled_cloud.points.emplace_back(sampled_point.x(), sampled_point.y(), sampled_point.z());
    }

    *cloud = sampled_cloud;
    cloud->width = cloud->points.size();
    cloud->height = 1; // 表示无序点云
    cloud->is_dense = true;
    log("Point cloud sampled. Size: " + std::to_string(cloud->points.size()));
}

// 体素下采样
void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, float leaf_size) {
    // 确保 cloud 和 cloud_filtered 不为空
    if (!cloud || !cloud_filtered) {
        log("Invalid input or output cloud pointers.");
        return;
    }
    // 必须用PointCloud2来体素滤波，不然会有内存管理方面的错误。
    pcl::PCLPointCloud2::Ptr cloud2 (new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr cloud2_filtered (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2(*cloud, *cloud2);

    // 初始化体素栅格滤波器
    pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
    voxel_grid.setInputCloud(cloud2);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);

    // 执行滤波
    voxel_grid.filter(*cloud2_filtered);
    pcl::fromPCLPointCloud2(*cloud2_filtered, *cloud_filtered);

    // 确保点云尺寸设置正确
    cloud_filtered->width = static_cast<uint32_t>(cloud_filtered->points.size());
    cloud_filtered->height = 1; // 表示无序点云
    cloud_filtered->is_dense = true;
    log("Filtered point cloud size: " + std::to_string(cloud_filtered->points.size()));
}


// 最远点滤波
void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered, int num_samples) {
    log("Filtering point cloud using optimized Farthest Point Sampling with KD-Tree and " + std::to_string(num_samples) + " samples.");

    // 如果点云数量小于或等于采样点数，直接返回原始点云
    if (cloud->points.size() <= num_samples) {
        *cloud_filtered = *cloud;
        return;
    }

    // 初始化采样点索引
    std::vector<int> sampled_indices;
    sampled_indices.reserve(num_samples);

    // 随机选择第一个采样点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, cloud->points.size() - 1);
    sampled_indices.push_back(dis(gen));

    // 初始化最小距离数组，初始值为最大浮点数
    std::vector<float> min_distances(cloud->points.size(), std::numeric_limits<float>::max());

    // 初始化KD树
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    std::mutex mutex;

    for (int i = 1; i < num_samples; ++i) {
        int last_sampled_index = sampled_indices.back();
        pcl::PointXYZ last_sampled_point = cloud->points[last_sampled_index];

        // 更新最小距离
        std::vector<std::thread> threads;
        int num_threads = std::thread::hardware_concurrency();
        int chunk_size = cloud->points.size() / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                int start_idx = t * chunk_size;
                int end_idx = (t == num_threads - 1) ? cloud->points.size() : start_idx + chunk_size;
                for (int j = start_idx; j < end_idx; ++j) {
                    float distance = pcl::euclideanDistance(cloud->points[j], last_sampled_point);
                    if (distance < min_distances[j]) {
                        min_distances[j] = distance;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // 查找最远的点
        float max_distance = -1.0f;
        int farthest_index = -1;

        threads.clear();
        std::vector<float> local_max_distances(num_threads, -1.0f);
        std::vector<int> local_farthest_indices(num_threads, -1);

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                int start_idx = t * chunk_size;
                int end_idx = (t == num_threads - 1) ? cloud->points.size() : start_idx + chunk_size;
                for (int j = start_idx; j < end_idx; ++j) {
                    if (min_distances[j] > local_max_distances[t]) {
                        local_max_distances[t] = min_distances[j];
                        local_farthest_indices[t] = j;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (int t = 0; t < num_threads; ++t) {
            if (local_max_distances[t] > max_distance) {
                max_distance = local_max_distances[t];
                farthest_index = local_farthest_indices[t];
            }
        }

        // 添加最远的点到采样索引中
        sampled_indices.push_back(farthest_index);
    }

    // 根据采样索引生成采样后的点云
    cloud_filtered->points.resize(sampled_indices.size());
    for (size_t i = 0; i < sampled_indices.size(); ++i) {
        cloud_filtered->points[i] = cloud->points[sampled_indices[i]];
    }

    // 设置点云的尺寸和密度
    cloud_filtered->width = static_cast<uint32_t>(cloud_filtered->points.size());
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;

    log("Filtered point cloud size: " + std::to_string(cloud_filtered->points.size()));
}

// 泊松圆盘滤波
void filterPointCloud2(const pcl::PCLPointCloud2::Ptr& cloud_in, pcl::PCLPointCloud2::Ptr& cloud_filtered, float radius) {
    log("Starting Poisson Disk Sampling with radius: " + std::to_string(radius));

    // 提取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*cloud_in, *cloud_xyz);

    // 构建KD树
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_xyz);

    std::vector<int> sampled_indices;
    std::vector<bool> sampled(cloud_xyz->points.size(), false);

    std::mutex mtx;

    // 采样函数
    auto sampleFunction = [&](int start_idx, int end_idx) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            if (sampled[idx]) continue;

            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            pcl::PointXYZ search_point = cloud_xyz->points[idx];

            if (kdtree.radiusSearch(search_point, radius, nn_indices, nn_dists) > 0) {
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    for (int nn_idx : nn_indices) {
                        sampled[nn_idx] = true;
                    }
                    sampled_indices.push_back(idx);
                }
            }
        }
    };

    // 创建多线程进行并行处理
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int chunk_size = cloud_xyz->points.size() / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start_idx = i * chunk_size;
        int end_idx = (i == num_threads - 1) ? cloud_xyz->points.size() : start_idx + chunk_size;
        threads.emplace_back(sampleFunction, start_idx, end_idx);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // 提取采样点云
    pcl::PointCloud<pcl::PointXYZ> sampled_points;
    for (int idx : sampled_indices) {
        sampled_points.points.push_back(cloud_xyz->points[idx]);
    }

    // 转换为PCLPointCloud2格式
    pcl::toPCLPointCloud2(sampled_points, *cloud_filtered);
    log("Filtered point cloud size: " + std::to_string(sampled_points.points.size()));
}


// 将点云保存到TXT文件和PCD文件
void savePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& txt_filename) {

    // 保存到TXT文件
    std::ofstream txt_file(txt_filename);
    if (!txt_file.is_open()) {
        log("Failed to open TXT file: " + txt_filename);
        return;
    }
    for (const auto& point : cloud->points) {
        txt_file << point.x << " " << point.y << " " << point.z << std::endl;
    }
    txt_file.close();
    log("Point cloud saved to TXT file: " + txt_filename);

}

// 计算点云中粒子距离
void com_dis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    double total_min_distance = 0.0;
    std::vector<double> distance;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        std::vector<int> pointIdxNKNSearch(2);
        std::vector<float> pointNKNSquaredDistance(2);

        // KNN搜索，寻找最近的一个邻居点
        if (kdtree.nearestKSearch(cloud->points[i], 2, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            total_min_distance += sqrt(pointNKNSquaredDistance[1]);
            distance.push_back(sqrt(pointNKNSquaredDistance[1]));
        }
    }
    double average_min_distance = total_min_distance / cloud->points.size();
    double var_total = 0.0;
    for(double i : distance) {
        var_total += (i - average_min_distance) * (i - average_min_distance);
    }
    double var = var_total / distance.size();
    log("Average minimum distance: " + std::to_string(average_min_distance));
    log("Var: " + std::to_string(var));
}

// K-means++初始化函数
std::vector<unsigned int> kMeansPlusPlusInit(const std::vector<pcl::Kmeans::Point>& data, int k) {
    std::vector<unsigned int> centers;
    std::random_device rd;
    std::mt19937 gen(rd());

    // 随机选择第一个中心
    std::uniform_int_distribution<> dis(0, data.size() - 1);
    centers.push_back(dis(gen));

    // 选择剩余的中心
    for (int i = 1; i < k; ++i) {
        std::vector<float> dist(data.size(), std::numeric_limits<float>::max());

        for (size_t j = 0; j < data.size(); ++j) {
            for (int center : centers) {
                float d = 0.0f;
                for (size_t dim = 0; dim < data[j].size(); ++dim) {
                    float diff = data[j][dim] - data[center][dim];
                    d += diff * diff;
                }
                if (d < dist[j]) {
                    dist[j] = d;
                }
            }
        }

        std::discrete_distribution<> weighted_dis(dist.begin(), dist.end());
        centers.push_back(weighted_dis(gen));
    }

    return centers;
}

// 使用K-means++聚类算法生成多层点云的函数
void generateMultiLayerPointCloudWithKMeansPlusPlus(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
                                                    int num_layers, float scale_factor, int num_clusters) {
    // 计时器
    pcl::console::TicToc tt;
    tt.tic();

    // 准备数据集
    std::vector<pcl::Kmeans::Point> data(input_cloud->points.size(), pcl::Kmeans::Point(3));
    for (size_t i = 0; i < input_cloud->points.size(); ++i) {
        data[i][0] = input_cloud->points[i].x;
        data[i][1] = input_cloud->points[i].y;
        data[i][2] = input_cloud->points[i].z;
    }

    // K-means++聚类
    pcl::Kmeans kmeans(data.size(), 3);
    kmeans.setClusterSize(num_clusters);
    std::vector<unsigned int> centers = kMeansPlusPlusInit(data, num_clusters);
    kmeans.setInputData(data);
    kmeans.kMeans();

    log("K-means++ clustering completed in " + std::to_string(tt.toc()) + " ms.");

    // 获取聚类中心
    auto centroids = kmeans.get_centroids();

    // 将点分配到各自的聚类
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters(num_clusters);
    for (int i = 0; i < num_clusters; ++i) {
        clusters[i].reset(new pcl::PointCloud<pcl::PointXYZ>);
    }

    auto points_to_clusters = kmeans.getPointsToClusters();
    for (size_t i = 0; i < input_cloud->points.size(); ++i) {
        int cluster_id = points_to_clusters[i];
        clusters[cluster_id]->points.push_back(input_cloud->points[i]);
    }

    output_cloud->clear();
    std::mutex cloud_mutex;

    // 并行处理每个聚类
    auto process_cluster = [&](int cluster_idx) {
        const auto& cluster = clusters[cluster_idx];
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster, centroid);

        for (int layer = 0; layer < num_layers; ++layer) {
            float current_scale = pow(scale_factor, layer);

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& point : cluster->points) {
                pcl::PointXYZ scaled_point;
                scaled_point.x = centroid[0] + (point.x - centroid[0]) * current_scale;
                scaled_point.y = centroid[1] + (point.y - centroid[1]) * current_scale;
                scaled_point.z = centroid[2] + (point.z - centroid[2]) * current_scale;
                transformed_cloud->points.push_back(scaled_point);
            }

            // 设置transformed_cloud的宽度和高度
            transformed_cloud->width = transformed_cloud->points.size();
            transformed_cloud->height = 1;
            transformed_cloud->is_dense = true;

            std::lock_guard<std::mutex> lock(cloud_mutex);
            *output_cloud += *transformed_cloud;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_clusters; ++i) {
        threads.emplace_back(process_cluster, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // 设置输出点云的宽度和高度
    output_cloud->width = output_cloud->points.size();
    output_cloud->height = 1;
    output_cloud->is_dense = true;

    log("final point cloud size: " + std::to_string(output_cloud->points.size()));
    log("Multi-layer point cloud generation completed.");
}


// 可视化点云
void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {

    pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) {
        // 等待窗口关闭
    }
}

}
}
}