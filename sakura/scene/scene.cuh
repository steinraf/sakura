#pragma once

#include <omp.h>

#include <chrono>
#include <fstream>
#include <optional>
#include <vector>
#include <iostream>

#include "imgui.h"

#include "../geometry/triangle.cuh"
#include "../camera/camera.cuh"
#include "pngwriter.h"


void checkCudaErrors(cudaError result);

__global__ void render_kern(Triangle *triangles, size_t triangleCount,
                            cudaSurfaceObject_t surface,
                            Eigen::Transform<float, 3, Eigen::Affine> cameraTf,
                            int width, int height);


class Scene {

public:
    void render(cudaSurfaceObject_t surface, Camera& camera, const ImVec2& windowSize) const;

private:
    friend class SceneBuilder;
    Scene() = default;
    Triangle *triangles;
    size_t triangleCount;
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

    SceneBuilder &setWorldToCamera(
            Eigen::Transform<float, 3, Eigen::Affine> tf);

    SceneBuilder &addTriangle(const Triangle &triangle);

    SceneBuilder &setWindowSize(int width, int height);

    [[nodiscard]] Scene build();

private:
    std::vector<Triangle> triangles;
    std::optional<Eigen::Vector2i> windowSize;
    std::optional<Eigen::Transform<float, 3, Eigen::Affine>> cameraTf;
};


