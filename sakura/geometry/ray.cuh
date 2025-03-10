//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>
#include <cuda/std/limits>

__device__ constexpr float RAY_EPSILON = 0.0001f;

struct Ray {
    __host__ __device__ Ray(Eigen::Vector3f origin, Eigen::Vector3f dir,
                            float minDist = RAY_EPSILON, float maxDist = cuda::std::numeric_limits<float>::infinity()) noexcept;

    [[nodiscard]] __host__ __device__ Eigen::Vector3f at(float t) const noexcept;

    __host__ __device__ void transform(const Eigen::Affine3f &transform) noexcept;


    Eigen::Vector3f origin;
    Eigen::Vector3f dir;
    float minDist, maxDist;
};