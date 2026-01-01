//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>
#include <cuda/std/limits>

#include "../common.cuh"

__device__ constexpr float RAY_EPSILON = 0.0001f;

struct Ray {
    CPU_GPU Ray(Vec3f origin, Vec3f dir,
                float minDist = RAY_EPSILON, float maxDist = cuda::std::numeric_limits<float>::infinity()) noexcept;

    [[nodiscard]] CPU_GPU Vec3f at(float t) const noexcept;

    CPU_GPU void transform(const Eigen::Affine3f &transform) noexcept;


    Vec3f origin;
    Vec3f dir;// Not necessarily unit vector (https://pbr-book.org/4ed/Shapes/Basic_Shape_Interface#IntersectionCoordinateSpaces)
    float minDist, maxDist;
};