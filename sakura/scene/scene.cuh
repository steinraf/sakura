#pragma once




#include <chrono>
#include <fstream>
#include <optional>
#include <vector>
#include <iostream>

#include <omp.h>
#include <curand_kernel.h>

#include "imgui.h"
#include "pngwriter.h"


#include "../geometry/triangle.cuh"
#include "../camera/camera.cuh"
#include "../acceleration/bvh.cuh"


void checkCudaErrors(cudaError result);


class Scene {

public:
    void render(cudaSurfaceObject_t surface, Camera& camera, curandState *rngStates, const ImVec2& windowSize, int spp) const;

private:
    friend class SceneBuilder;
    Scene() = default;
    BVH *bvh;
};

class SceneBuilder {
public:
    SceneBuilder() = default;
    SceneBuilder(const SceneBuilder &) = delete;
    SceneBuilder &operator=(const SceneBuilder &) = delete;

    SceneBuilder &addObj(
            const std::string &filename,
            Eigen::Transform<float, 3, Eigen::Affine> tf =
                    Eigen::Transform<float, 3, Eigen::Affine>::Identity());

    SceneBuilder &addTriangle(const Triangle &triangle);

    [[nodiscard]] Scene build();

private:
    std::vector<Triangle> triangles;
    std::optional<Eigen::Vector2i> windowSize;

};


__device__ __host__ constexpr uint32_t LeftShift3(uint32_t x) noexcept;