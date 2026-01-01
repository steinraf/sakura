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
    Vec3f min, max;

    CPU_GPU AABB() noexcept;

    CPU_GPU AABB(Vec3f min, Vec3f max) noexcept;

    CPU_GPU AABB(const Vec3f &p0,
                 const Vec3f &p1,
                 const Vec3f &p2) noexcept;

    [[nodiscard]] CPU_GPU bool intersect(
            const Ray &ray) const noexcept;

    [[nodiscard]] CPU_GPU AABB
    operator+(const AABB &other) const noexcept;

    [[nodiscard]] CPU_GPU Vec3f getCenter()
            const noexcept {
        return (min + max) / 2;
    }

    // TODO handle degenerate AABBs where one dimension has width 0
    [[nodiscard]] CPU_GPU bool isFaulty() const noexcept;
};