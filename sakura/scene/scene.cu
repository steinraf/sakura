//
// Created by steinraf on 04.02.25.
//

#include "scene.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void checkCudaErrors(cudaError result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
}

__global__ void render_kern(BVH *bvh, cudaSurfaceObject_t surface, Camera camera, int width, int height){

    for (unsigned int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height;
        pixelIndex += blockDim.x * gridDim.x) {

        unsigned int x = pixelIndex % width, y = height - 1 - pixelIndex / width;

        auto color = Eigen::Vector3f{0.0, 0.0, 0.0};
        auto screenPos = Eigen::Vector3f{x * 1.f / width,
                                         y * 1.f / height, 0.0f};


        auto ray =  camera.getRay(float(x) / width, float(y) / height) ;

        Intersection intersection;
        auto t = Eigen::Vector3f{1.0, 1.0, 1.0};
        const Eigen::Vector3f backgroundColor =
                1.f * Eigen::Vector3f{1.0, 0.0, 0.0};

        while(true){
            if (!bvh->intersect(ray, intersection)) {
                color.array() += t.array() * backgroundColor.array();
                break;
            }



            const auto &triangle = *intersection.triangle;

            const Eigen::Vector3f bary = {
                    1.0f - intersection.uv[0] - intersection.uv[1],
                    intersection.uv[0], intersection.uv[1]};


            auto normal = Eigen::Vector3f{bary[0] * triangle.n0 +
                                          bary[1] * triangle.n1 +
                                          bary[2] * triangle.n2};

            float orientation = abs(normal.dot(ray.dir));
            color = Eigen::Vector3f{orientation, orientation,
                                    orientation};

            ray.maxDist = intersection.t;

            break;
        }

        uchar4 color4 = make_uchar4(color[0] * 255, color[1] * 255, color[2] * 255, 255);
        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}
void Scene::render(cudaSurfaceObject_t surface, Camera& camera, const ImVec2& windowSize) const {


    int devId = 0;
    int numSMs;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));


    auto start = std::chrono::high_resolution_clock::now();

    render_kern<<<32 * numSMs, 256>>>(bvh, surface, camera,  windowSize[0], windowSize[1]);

    checkCudaErrors(cudaDeviceSynchronize());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
}


SceneBuilder &SceneBuilder::addObj(
        const std::string &filename, Eigen::Transform<float, 3, Eigen::Affine> tf) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<Eigen::Vector3f> vertices{};
    std::vector<Eigen::Vector3f> normals{};

    std::vector<Triangle> triangles{};

    std::string lineString{};

    while (std::getline(file, lineString)) {
        std::istringstream line{lineString};
        std::string start;

        line >> start;

        if (start == "v") {
            Eigen::Vector3f vertex;
            line >> vertex.x() >> vertex.y() >> vertex.z();
            vertices.push_back(vertex);
        } else if (start == "vn") {
            Eigen::Vector3f normal;
            line >> normal.x() >> normal.y() >> normal.z();
            normals.push_back(normal);
        } else if (start == "f") {
            std::array<Eigen::Vector3f, 3> faceVertices;
            std::array<Eigen::Vector3f, 3> faceNormals;

            for (int i = 0; i < 3; i++) {
                std::string vertex;
                line >> vertex;
                std::istringstream vertexStream{vertex};
                std::string vertexIndex;
                std::getline(vertexStream, vertexIndex, '/');
                faceVertices[i] = vertices[std::stoi(vertexIndex) - 1];
                faceVertices[i][2] *= -1; //TODO remove
                std::string textureIndex;
                std::getline(vertexStream, textureIndex, '/');
                //                uvTextures[i] = uvs[std::stoi(textureIndex) -
                //                1];
                std::string normalIndex;
                std::getline(vertexStream, normalIndex, '/');
                faceNormals[i] = normals[std::stoi(normalIndex) - 1];
            }



            triangles.emplace_back(
                    tf * faceVertices[0], tf * faceVertices[1],
                    tf * faceVertices[2], tf.linear() * faceNormals[0],
                    tf.linear() * faceNormals[1], tf.linear() * faceNormals[2]);
        }
    }
    for (const auto &t : triangles) {
        addTriangle(t);
    }
    return *this;
}

SceneBuilder &SceneBuilder::addTriangle(const Triangle &triangle) {
    triangles.push_back(triangle);
    return *this;
}

Scene SceneBuilder::build() {
    auto scene = Scene{};

    Triangle *trias;
    checkCudaErrors(
            cudaMallocManaged(&trias, triangles.size() * sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(trias, triangles.data(),
                               triangles.size() * sizeof(Triangle),
                               cudaMemcpyHostToDevice));

    std::cout << "Allocated Triangles\n";

    AABB boundingBox = thrust::transform_reduce(
            thrust::device, trias, trias + triangles.size(),
            [=] __host__ __device__(const Triangle &t) -> AABB {
                return t.AABBGetter();
            },
            AABB{}, thrust::plus<AABB>());

    std::cout << "Scene Bounding Box: Min\n"
              << boundingBox.min << "\nMax\n"
              << boundingBox.max << '\n';

    Eigen::Vector3f lower = boundingBox.min;
    Eigen::Vector3f dims = boundingBox.max - boundingBox.min;

    thrust::device_vector<uint32_t> mortonCodes(triangles.size());
    thrust::transform(
            thrust::device, trias, trias + triangles.size(), mortonCodes.begin(),
            [=] __host__ __device__(const Triangle &tria) {
                int numBits = 10;
                const Eigen::Vector3f normalized =
                        static_cast<float>(1u << numBits) *
                        (tria.AABBGetter().getCenter() - lower).array() / dims.array();

                assert(normalized[0] >= 0 && normalized[1] >= 0 &&
                       normalized[2] >= 0);
                assert(normalized[0] <= (1u << numBits) &&
                       normalized[1] <= (1u << numBits) &&
                       normalized[2] <= (1u << numBits));

                return (LeftShift3(static_cast<uint32_t>(normalized[2])) << 2) |
                       (LeftShift3(static_cast<uint32_t>(normalized[1])) << 1) |
                       (LeftShift3(static_cast<uint32_t>(normalized[0])));
            });

    thrust::sort_by_key(thrust::device, mortonCodes.begin(), mortonCodes.end(),
                        trias);

    std::cout << "Sorted " << triangles.size() << " Triangles by Morton Code\n";
    BVH *bvh;
    AccelerationNode *bvhNodes;

    checkCudaErrors(cudaMallocManaged(&bvh, sizeof(BVH)));
    checkCudaErrors(cudaMallocManaged(
            &bvhNodes, 2 * triangles.size() * sizeof(AccelerationNode)));

    unsigned short int *bvhConstructionDone;
    checkCudaErrors(
            cudaMallocManaged(&bvhConstructionDone,
                              (triangles.size() - 2) * sizeof(unsigned short int)));

    auto start = std::chrono::high_resolution_clock::now();
    // TODO grid stride loop or tune sizes
    constructBVH<<<(triangles.size() + 255) / 256, 256>>>(
            bvhNodes, trias, triangles.size(), mortonCodes.data().get(),
            bvhConstructionDone);

    checkCudaErrors(cudaDeviceSynchronize());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

    std::cout << "Built BVH in " << duration.count() << "ms\n";

    *bvh = BVH{bvhNodes, triangles.size()};

    scene.bvh = bvh;

//    scene.triangles = trias;
//    scene.triangleCount = triangles.size();

//    if (cameraTf.has_value()) {
//        throw std::runtime_error("Doesnt allow for camera in scene. Add to Viewport instead.");
//    }

    if (windowSize.has_value()) {
        throw std::runtime_error("Doesnt allow for window size in scene. Add to Viewport instead.");
    }

    return scene;
}

__device__ __host__ constexpr uint32_t LeftShift3(uint32_t x) noexcept {
    if (x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    return x;
}