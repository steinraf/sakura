//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "../utility/ray.h"
#include "../utility/vector.cuh"

#include <cuda/std/limits>


struct AABB {

    Vector3f min, max;


    __host__ __device__ constexpr AABB() noexcept
        : min(INFINITY), max(-INFINITY) {
    }

    __host__ __device__ constexpr AABB(const Vector3f &min, const Vector3f max) noexcept
        : min(min), max(max) {
    }

    __host__ __device__ constexpr AABB(const Vector3f &a, const Vector3f &b, const Vector3f &c) noexcept
        : min({
                  CustomRenderer::min(a[0], CustomRenderer::min(b[0], c[0])),
                  CustomRenderer::min(a[1], CustomRenderer::min(b[1], c[1])),
                  CustomRenderer::min(a[2], CustomRenderer::min(b[2], c[2])),
          }),
          max({
                  CustomRenderer::max(a[0], CustomRenderer::max(b[0], c[0])),
                  CustomRenderer::max(a[1], CustomRenderer::max(b[1], c[1])),
                  CustomRenderer::max(a[2], CustomRenderer::max(b[2], c[2])),
          }) {
    }

    //Nori BoundingBox RayIntersect
    [[nodiscard]] __device__ constexpr bool rayIntersect(const Ray3f &ray) const noexcept {
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

                nearT = CustomRenderer::max(t1, nearT);
                farT = CustomRenderer::min(t2, farT);

                if(nearT > farT)
                    return false;
            }
        }

        return ray.minDist <= farT && nearT <= ray.maxDist;
    }


    [[nodiscard]] __device__ __host__ constexpr inline Vector3f getCenter() const noexcept {
        return 0.5f * (min + max);
    }

    [[nodiscard]] __device__ constexpr inline bool isEmpty() const noexcept {
        return min == Vector3f(INFINITY) && max == Vector3f(-INFINITY);
    }

    [[nodiscard]] __device__ constexpr AABB operator+(const AABB &other) const noexcept {
        return {
                Vector3f{-EPSILON} + Vector3f{  CustomRenderer::min(min[0], other.min[0]),
                                                CustomRenderer::min(min[1], other.min[1]),
                                                CustomRenderer::min(min[2], other.min[2])},
                Vector3f{+EPSILON} + Vector3f{  CustomRenderer::max(max[0], other.max[0]),
                                                CustomRenderer::max(max[1], other.max[1]),
                                                CustomRenderer::max(max[2], other.max[2])},
        };
    }

};
