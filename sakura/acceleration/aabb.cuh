//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <thrust/extrema.h>

#include <Eigen/Dense>
#include <cuda/std/limits>

#include "../common.cuh"

// Axis Aligned Bounding Box
struct AABB {
    Eigen::Vector3f min, max;

    __host__ __device__ AABB() noexcept;

    __host__ __device__ AABB(Eigen::Vector3f min, Eigen::Vector3f max) noexcept;

    __host__ __device__ AABB(const Eigen::Vector3f &p0,
                             const Eigen::Vector3f &p1,
                             const Eigen::Vector3f &p2) noexcept;

    [[nodiscard]] __host__ __device__ bool intersect(
            const Ray &ray) const noexcept;

    [[nodiscard]] __host__ __device__ AABB
    operator+(const AABB &other) const noexcept;

    [[nodiscard]] __host__ __device__ Eigen::Vector3f getCenter()
            const noexcept {
        return (min + max) / 2;
    }

    // TODO handle degenerate AABBs where one dimension has width 0
    [[nodiscard]] __host__ __device__ bool isFaulty() const noexcept;
};