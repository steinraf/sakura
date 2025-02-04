//
// Created by steinraf on 04.02.25.
//

#include "scene.cuh"

void checkCudaErrors(cudaError result){
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
}

__global__ void render_kern(Triangle *triangles, size_t triangleCount, cudaSurfaceObject_t surface, Eigen::Transform<float, 3, Eigen::Affine> cameraTf, int width, int height){

    for (unsigned int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height;
        pixelIndex += blockDim.x * gridDim.x) {

        unsigned int x = pixelIndex % width, y = height - 1 - pixelIndex / width;

        auto color = Eigen::Vector3f{1.0, 1.0, 1.0};
        auto screenPos = Eigen::Vector3f{x * 1.f / width,
                                         y * 1.f / height, 0.0f};


        auto ray =  Ray{cameraTf * screenPos,
                    cameraTf.linear() * Eigen::Vector3f{-screenPos[0] + 0.5f, screenPos[1] - 0.5f, 1.f}
                           .normalized()} ;

        Intersection intersection;
        for (int triangle_idx = 0; triangle_idx < triangleCount;
            ++triangle_idx) {
            const auto &triangle = triangles[triangle_idx];
            if (triangle.intersect(ray, intersection)) {

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
            }
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

    render_kern<<<32 * numSMs, 256>>>(triangles, triangleCount, surface, camera.tf,  windowSize[0], windowSize[1]);

    checkCudaErrors(cudaDeviceSynchronize());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
}


SceneBuilder &SceneBuilder::addObj(const std::string &filename, Eigen::Transform<float, 3, Eigen::Affine> tf) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<Eigen::Vector3f> vertices{};
    std::vector<Eigen::Vector3f> normals{};

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
                std::string normalIndex;
                std::getline(vertexStream, normalIndex, '/');
                faceNormals[i] = normals[std::stoi(normalIndex) - 1];
            }

            addTriangle(Triangle{tf * faceVertices[0], tf * faceVertices[1],
                                 tf * faceVertices[2],
                                 tf.linear() * faceNormals[0],
                                 tf.linear() * faceNormals[1],
                                 tf.linear() * faceNormals[2]});
        }
    }
    return *this;
}
SceneBuilder &SceneBuilder::setWorldToCamera(Eigen::Transform<float, 3, Eigen::Affine> tf) {
    cameraTf = tf;
    return *this;
}
SceneBuilder &SceneBuilder::addTriangle(const Triangle &triangle) {
    triangles.push_back(triangle);
    return *this;
}
SceneBuilder &SceneBuilder::setWindowSize(int width, int height) {
    this->windowSize = Eigen::Vector2i{width, height};
    return *this;
}
Scene SceneBuilder::build() {
    auto scene = Scene{};

    Triangle *trias;
    checkCudaErrors(cudaMallocManaged(&trias, triangles.size() * sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(trias, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

    scene.triangles = trias;
    scene.triangleCount = triangles.size();

    if (cameraTf.has_value()) {
        throw std::runtime_error("Doesnt allow for camera in scene. Add to Viewport instead.");
    }

    if (windowSize.has_value()) {
        throw std::runtime_error("Doesnt allow for window size in scene. Add to Viewport instead.");
    }

    return scene;
}

