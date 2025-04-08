//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "../utility/ray.h"
#include "../utility/vector.cuh"

#include <cuda/std/limits>


struct AABB {

    Vector3f min, max;


    CPU_GPU_CONSTEXPR AABB() noexcept
        : min(INFINITY), max(-INFINITY) {
    }

    CPU_GPU_CONSTEXPR AABB(const Vector3f &min, const Vector3f max) noexcept
        : min(min), max(max) {
    }

    CPU_GPU_CONSTEXPR AABB(const Vector3f &a, const Vector3f &b, const Vector3f &c) noexcept
        : min({
                  std::min(a[0], std::min(b[0], c[0])),
                  std::min(a[1], std::min(b[1], c[1])),
                  std::min(a[2], std::min(b[2], c[2])),
          }),
          max({
                  std::max(a[0], std::max(b[0], c[0])),
                  std::max(a[1], std::max(b[1], c[1])),
                  std::max(a[2], std::max(b[2], c[2])),
          }) {
    }

    //Nori BoundingBox RayIntersect
    [[nodiscard]] CPU_GPU_CONSTEXPR bool rayIntersect(const Ray3f &ray) const noexcept {
        float nearT = -cuda::std::numeric_limits<float>::infinity();
        float farT = cuda::std::numeric_limits<float>::infinity();

        for(int i = 0; i < 3; i++) {
            float origin = ray.getOrigin()[i];
            float minVal = min[i], maxVal = max[i];

            if(ray.getDirection()[i] == 0) {
                if(origin < minVal || origin > maxVal)
                    return false;
            } else {
                float t1 = (minVal - origin) / ray.getDirection()[i];
                float t2 = (maxVal - origin) / ray.getDirection()[i];

                if(t1 > t2) {
                    cuda::std::swap(t1, t2);
                }

                nearT = std::max(t1, nearT);
                farT = std::min(t2, farT);

                if(nearT > farT)
                    return false;
            }
        }

        return ray.getMinDist() <= farT && nearT <= ray.getMaxDist();
    }


    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f getCenter() const noexcept {
        return 0.5f * (min + max);
    }

    [[nodiscard]] CPU_GPU_CONSTEXPR bool isEmpty() const noexcept {
        return min == Vector3f(INFINITY) && max == Vector3f(-INFINITY);
    }

    [[nodiscard]] CPU_GPU_CONSTEXPR AABB operator+(const AABB &other) const noexcept {
        return {
                Vector3f{-EPSILON} + Vector3f{  std::min(min[0], other.min[0]),
                                                std::min(min[1], other.min[1]),
                                                std::min(min[2], other.min[2])},
                Vector3f{+EPSILON} + Vector3f{  std::max(max[0], other.max[0]),
                                                std::max(max[1], other.max[1]),
                                              std::max(max[2], other.max[2])},
        };
    }

};
